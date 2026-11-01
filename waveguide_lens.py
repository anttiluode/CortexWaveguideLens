"""CortexWaveguideLens: tiny, falsifiable phase-steered memory lens.

This is SIGNAL PROCESSING, not a biophysical neuron model. All context is
observed before the query. No labels, true shifts, or corrupt-band flags enter
models during inference.

Dependencies: numpy, torch (CPU sufficient).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N = 64
BANDS = np.array([3, 5, 7, 11, 14, 19], dtype=np.int64)
SHIFTS = np.arange(N)


def generate(count: int, seed: int, shifts: np.ndarray, noise_scale: float = 0.85):
    """Generate reference, independent noise probe, and later query.

    Negative query has the same clean Fourier magnitudes as the reference but
    independent phases. Positive query is an unknown circular translation.
    Both have identically distributed corruptions and probe measurements.
    The independently observed noise probe indicates which bands are noisy.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, count)
    shift = rng.choice(shifts, count)
    ref_phase = rng.uniform(-np.pi, np.pi, (count, len(BANDS)))
    false_phase = rng.uniform(-np.pi, np.pi, (count, len(BANDS)))
    # Two independently varying environments: low bands or high bands noisy.
    environment = rng.integers(0, 2, count)
    high = np.where(environment[:, None] == 0, np.arange(len(BANDS))[None, :] < 3,
                    np.arange(len(BANDS))[None, :] >= 3)
    # Low/high reversed by context, with per-sample stochastic variations.
    sigma = noise_scale * np.where(high, 1.9, .18) * rng.lognormal(0, .18, (count, len(BANDS)))
    # Reference amplitude varies, but same for both labels. Magnitudes by
    # themselves reveal no class. Inverse FFT creates real-valued waveforms.
    amp = rng.uniform(.75, 1.25, (count, len(BANDS)))
    zref = amp * np.exp(1j * ref_phase)
    clean_phase = np.where(labels[:, None] == 1, ref_phase, false_phase)
    zclean = amp * np.exp(1j * (clean_phase - 2*np.pi*shift[:, None]*BANDS[None, :]/N))
    noise = sigma * (rng.normal(size=zclean.shape) + 1j*rng.normal(size=zclean.shape)) / np.sqrt(2)
    zquery = zclean + noise
    # 8 probes, giving a noisy (not oracle) estimate of the environment.
    probes = sigma[:, None, :] * (rng.normal(size=(count,8,len(BANDS))) +
                     1j * rng.normal(size=(count,8,len(BANDS)))) / np.sqrt(2)
    pwr = (np.abs(probes)**2).mean(axis=1)
    # Materialize real sensor waveforms. The lens must recover its frequency
    # bands from those observations; it does not receive generating phases.
    def to_wave(z):
        f = np.zeros((count, N//2+1), dtype=np.complex128)
        f[:, BANDS] = z
        return np.fft.irfft(f, n=N, axis=1).astype(np.float32)
    return {'labels':labels.astype(np.float32), 'shift':shift,
            'ref':to_wave(zref), 'query':to_wave(zquery), 'probe':np.log(pwr+1e-5).astype(np.float32),
            'sigma':sigma, 'environment':environment}


def phase_features(examples):
    """One complex channel per band; candidate 'cable delays' across 64 shifts."""
    zr = np.fft.rfft(examples['ref'],n=N,axis=1)[:,BANDS]
    zq = np.fft.rfft(examples['query'],n=N,axis=1)[:,BANDS]
    p = zq * zr.conj() / (np.abs(zq)*np.abs(zr)+1e-8)
    # Positive samples have angle(zq*zref*) = -omega*translation.
    rot = np.exp(2j * np.pi * SHIFTS[:,None] * BANDS[None,:] / N)
    coherence = (p[:,None,:] * rot[None,:,:]).real.astype(np.float32)
    # Fixed comparator at zero delay, instead of scanning all candidate delays.
    magnitude = np.stack([np.log(np.abs(zr)+1e-5),
                          np.log(np.abs(zq)+1e-5)],axis=-1).astype(np.float32)
    return coherence, magnitude


class BiasOnlyLens(nn.Module):
    """Matched access to the noise probe, but NOT to channel-specific routing."""
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(len(BANDS)))
        self.bias_router = nn.Sequential(nn.Linear(len(BANDS), 16),nn.Tanh(),
                                         nn.Linear(16, 1))
        self.slope = nn.Parameter(torch.tensor(7.0))
        self.bias = nn.Parameter(torch.tensor(-3.5))

    def forward(self, cosines, probe):
        weights=torch.softmax(self.logits,dim=0)
        score=(cosines*weights[None,None,:]).sum(-1).max(1).values
        return self.slope*score + self.bias + self.bias_router(probe).squeeze(-1)


class AdaptiveLens(nn.Module):
    """A prior state (noise probe) modulates branch gains before coherent readout."""
    def __init__(self, adaptive: bool = True):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.router = nn.Sequential(nn.Linear(len(BANDS), 16), nn.Tanh(),
                                        nn.Linear(16, len(BANDS)))
        else:
            self.logits = nn.Parameter(torch.zeros(len(BANDS)))
        self.slope = nn.Parameter(torch.tensor(7.0))
        self.bias = nn.Parameter(torch.tensor(-3.5))

    def forward(self, cosines, probe):
        if self.adaptive:
            logits = self.router(probe)
        else:
            logits = self.logits[None,:].expand(probe.size(0),-1)
        weights = torch.softmax(logits,dim=-1)
        # Channel gains are positive; phase rotations happen in cosines.
        coherent_score = (cosines * weights[:,None,:]).sum(dim=-1).max(dim=1).values
        return self.slope * coherent_score + self.bias


def train(model, cosines, probe, labels, steps=450, seed=0):
    torch.manual_seed(seed)
    x = torch.from_numpy(cosines)
    p = torch.from_numpy(probe)
    y = torch.from_numpy(labels)
    opt = torch.optim.Adam(model.parameters(), lr=0.014)
    rng = np.random.default_rng(seed+78)
    for _ in range(steps):
        ids = torch.from_numpy(rng.integers(0,len(y),size=192))
        opt.zero_grad()
        output = model(x[ids],p[ids])
        loss = F.binary_cross_entropy_with_logits(output,y[ids])
        loss.backward()
        opt.step()
    return model


def accuracy(logits, y):
    p = 1/(1+np.exp(-np.clip(logits,-40,40)))
    return float(((p>=0.5)==y).mean()), float(np.mean((p-y)**2))


def run(seed=3, steps=350, noise_scale=.85, test_noise_scale=None, train_size=2500, test_size=1500):
    # Training has shifts 0..31; test contains *unseen* shifts 32..63.
    a = generate(train_size,seed,SHIFTS[:32],noise_scale)
    b = generate(test_size,seed+1000,SHIFTS[32:],noise_scale if test_noise_scale is None else test_noise_scale)
    ca,ma=phase_features(a); cb,mb=phase_features(b)
    results = {}
    for key in ['fixed_gain','context_bias_only','apical_probe']:
        adaptive = key=='apical_probe'
        torch.manual_seed(seed+int(adaptive))
        model=train(BiasOnlyLens() if key=='context_bias_only' else AdaptiveLens(adaptive),ca,a['probe'],a['labels'],steps,seed)
        model.eval()
        with torch.no_grad():
            score=model(torch.from_numpy(cb),torch.from_numpy(b['probe'])).numpy()
            wrong_probe=b['probe'][np.random.default_rng(seed+456).permutation(len(b['labels']))]
            shuffled=model(torch.from_numpy(cb),torch.from_numpy(wrong_probe)).numpy()
            weights=(torch.softmax(model.router(torch.from_numpy(b['probe'])),dim=-1).numpy()
                     if adaptive else None)
        acc,brier=accuracy(score,b['labels'])
        results[key]={'accuracy':acc,'brier':brier, 'parameters':sum(v.numel() for v in model.parameters())}
        if adaptive:
            results['shuffled_apical_probe']={'accuracy':accuracy(shuffled,b['labels'])[0],
                                              'brier':accuracy(shuffled,b['labels'])[1]}
            results['learned_gains']={'low_bands_corrupted':weights[b['environment']==0].mean(axis=0).round(3).tolist(),
                                      'high_bands_corrupted':weights[b['environment']==1].mean(axis=0).round(3).tolist()}
    # Existing classical Fourier phase correlation / cross-correlation has the
    # same delay scan as our 'waveguide': don't claim this is novel mathematics.
    fixed_prob=cb.mean(axis=-1).max(axis=1)
    # Classical analytic control: estimate noise power from the independent
    # calibration probe and use inverse-power gains instead of a learned router.
    def classical_gate(c, logpower):
        weights=1.0/(np.exp(logpower)+.1)
        weights/=weights.sum(axis=1,keepdims=True)
        return (c*weights[:,None,:]).sum(-1).max(1)
    # A zero-delay comparator is deliberately handicapped w.r.t. translation.
    zero=cb[:,0,:].mean(axis=-1)
    # Pick thresholds on train ONLY (no test leakage).
    def optimize_threshold(train_score, test_score, key):
        thresholds=np.linspace(-.3,1.0,261)
        train_correct=[((train_score>t)==a['labels']).mean() for t in thresholds]
        th=float(thresholds[int(np.argmax(train_correct))])
        results[key]={'accuracy':float(((test_score>th)==b['labels']).mean()),'train_threshold':th}
    optimize_threshold(ca.mean(axis=-1).max(axis=1),fixed_prob,'uniform_phase_scan')
    optimize_threshold(classical_gate(ca,a['probe']),classical_gate(cb,b['probe']),'inverse_noise_classical')
    optimize_threshold(ca[:,0,:].mean(axis=-1),zero,'zero_delay')
    # Strict amplitude-only baseline: coefficients retain their magnitudes; no phase.
    # We train a small MLP on magnitude+probe, under same supervised labels.
    def mag_inputs(e,m): return np.concatenate([m.reshape(len(m),-1),e['probe']],axis=1).astype(np.float32)
    torch.manual_seed(seed+10)
    net=nn.Sequential(nn.Linear(3*len(BANDS),24),nn.Tanh(),nn.Linear(24,1))
    xa=torch.from_numpy(mag_inputs(a,ma)); ya=torch.from_numpy(a['labels'])
    xb=torch.from_numpy(mag_inputs(b,mb))
    opt=torch.optim.Adam(net.parameters(),lr=.004)
    rng=np.random.default_rng(seed+33)
    for _ in range(steps):
        ids=torch.from_numpy(rng.integers(0,len(ya),size=192))
        opt.zero_grad(); pred=net(xa[ids]).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(pred,ya[ids]);loss.backward();opt.step()
    with torch.no_grad(): magnitude_pred=net(xb).squeeze(1).numpy()
    results['magnitude_only']={'accuracy':accuracy(magnitude_pred,b['labels'])[0],
                               'brier':accuracy(magnitude_pred,b['labels'])[1]}
    # Memory shuffled in both train/test would make it truly uninformative.
    ar=np.random.default_rng(seed+74).permutation(len(a['labels']));br=np.random.default_rng(seed+75).permutation(len(b['labels']))
    shuffled_train={**a,'ref':a['ref'][ar]}
    shuffled_test={**b,'ref':b['ref'][br]}
    cs,_=phase_features(shuffled_train);ct,_=phase_features(shuffled_test)
    optimize_threshold(cs.mean(axis=-1).max(axis=1),ct.mean(axis=-1).max(axis=1),'shuffled_reference')
    results['metadata']={'seed':seed,'train':train_size,'test':test_size,'steps':steps,
                         'test_shifts':'32..63 (not in training)',
                         'train_shifts':'0..31', 'train_noise_scale':noise_scale,
                         'test_noise_scale':noise_scale if test_noise_scale is None else test_noise_scale,
                         'task':'phase-template matching under unknown translation and colored noise'}
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[3,17,27])
    ap.add_argument('--steps',type=int,default=350)
    ap.add_argument('--noise-scale',type=float,default=.85)
    ap.add_argument('--test-noise-scale',type=float,default=None)
    ap.add_argument('--output',default='')
    args=ap.parse_args()
    trials=[run(seed=s,steps=args.steps,noise_scale=args.noise_scale,
                test_noise_scale=args.test_noise_scale) for s in args.seeds]
    methods=['magnitude_only','zero_delay','shuffled_reference','uniform_phase_scan',
             'fixed_gain','context_bias_only','inverse_noise_classical',
             'apical_probe','shuffled_apical_probe']
    summary={method:{'mean_accuracy':round(float(np.mean([t[method]['accuracy'] for t in trials])),4),
                     'std_accuracy':round(float(np.std([t[method]['accuracy'] for t in trials],ddof=1)),4)}
             for method in methods}
    output={'summary':summary,'trials':trials}
    print(json.dumps(summary,indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':
    main()
