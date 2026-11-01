"""Stage 2: phase-propagated visual memory with causal uncertainty and controls.

This is an *estimator*, not a biological or learned wave medium. Known global
motion is supplied to all motion-compensated methods, making the benefit of
persistent phase-aligned state identifiable separately from motion estimation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

N = 48
T = 24


def image_scene(rng: np.random.Generator) -> np.ndarray:
    """Periodic structured scene containing localized oriented edges and texture."""
    yy, xx = np.mgrid[:N, :N]
    im = np.zeros((N, N), dtype=np.float64)
    for _ in range(7):
        cy, cx = rng.integers(0, N, size=2)
        sy, sx = rng.integers(3, 12, size=2)
        ydist = np.minimum((yy-cy)%N, (cy-yy)%N)
        xdist = np.minimum((xx-cx)%N, (cx-xx)%N)
        patch = ((ydist < sy) & (xdist < sx)).astype(float)
        im += rng.uniform(-1.0, 1.0) * gaussian_filter(patch, sigma=.45, mode='wrap')
    for _ in range(5):
        fy, fx = rng.integers(-8, 9, size=2)
        if fy == fx == 0: fx = 1
        im += rng.uniform(.10, .40)*np.cos(2*np.pi*(fx*xx+fy*yy)/N+rng.uniform(-np.pi,np.pi))
    im -= im.mean()
    im /= max(im.std(), 1e-8)
    return im


def phase_shift(im: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Exact periodic Fourier phase translation (also works for subpixels)."""
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.fftfreq(N)[None, :]
    phase = np.exp(-2j*np.pi*(ky*dy + kx*dx))
    return np.fft.ifft2(np.fft.fft2(im)*phase).real


def make_episode(seed: int, sigma: float=.55, switching: bool=True):
    rng=np.random.default_rng(seed)
    first=image_scene(rng)
    second=image_scene(rng)
    # Exactly one integer translation per frame. All methods get this velocity.
    dy, dx = [[0,1], [1,0], [1,-1], [-1,2]][int(rng.integers(0,4))]
    base = first
    clean, obs, masks = [], [], []
    for t in range(T):
        if switching and t == 13: base=second
        target = phase_shift(base, t*dy, t*dx)
        visible = np.ones((N,N), bool)
        # Two fixed-in-camera occlusions. The underlying object keeps moving.
        if 7 <= t <= 10 or 17 <= t <= 20:
            x0 = 12 if t <= 10 else 23
            visible[:,x0:x0+17] = False
        measurement = target + rng.normal(0, sigma, (N,N))
        measurement[~visible] = 0.0
        clean.append(target); obs.append(measurement); masks.append(visible)
    return np.stack(clean),np.stack(obs),np.stack(masks),(dy,dx)


def temporal_fixed(obs, mask, velocity, alpha=.30, motion=True):
    out=np.zeros_like(obs)
    previous=np.zeros_like(obs[0])
    for t in range(len(obs)):
        if t:
            previous = phase_shift(previous, *velocity) if motion else previous
        if t == 0:
            previous=np.where(mask[t], obs[t], 0)
        else:
            previous=np.where(mask[t], (1-alpha)*previous+alpha*obs[t],previous)
        out[t]=previous
    return out


def temporal_uncertain(obs, mask, velocity, sigma=.55, q=.005, reset_threshold=None,
                       motion=True, return_diagnostics=False):
    """Fourier phase-copy prediction plus measured innovation / uncertainty update.

    This is a simplified Kalman-style update (diagonal covariance). It is NOT
    claimed to be novel; no model parameters learn at inference time.
    """
    var=np.full(obs.shape[1:], sigma*sigma)
    state=np.zeros_like(obs[0]); out=np.zeros_like(obs)
    reset_times=[]; ratios=[]
    for t in range(T):
        if t:
            state=phase_shift(state,*velocity) if motion else state
            var=np.roll(var, shift=velocity, axis=(0,1)) if motion else var
            var += q
        if t == 0:
            state=np.where(mask[t],obs[t],0)
            var=np.where(mask[t],sigma*sigma,1e2)
        else:
            diff=obs[t]-state
            innov=np.mean(diff[mask[t]]**2/(sigma*sigma+var[mask[t]]))
            ratios.append(float(innov))
            if reset_threshold is not None and innov > reset_threshold:
                # Unexpected replacement: invalidate the old world prediction.
                state=np.where(mask[t],obs[t],0.)
                var=np.where(mask[t],sigma*sigma,1e2)
                reset_times.append(t)
            else:
                k=var/(var+sigma*sigma)
                state=np.where(mask[t],state+k*diff,state)
                var=np.where(mask[t],(1-k)*var,var)
        out[t]=state
    return (out,{'reset_times':reset_times,'innovation_ratios':ratios}) if return_diagnostics else out




def infer_velocity_from_first_two(obs, mask, max_step=2):
    """Estimate *one* constant velocity from observed frames 0 and 1 only.

    Each candidate is compared exclusively over pixels observed in both frames.
    This is a causal estimate, not scene-generation metadata.
    """
    best=float('inf'); best_vel=(0,0)
    for dy in range(-max_step,max_step+1):
        for dx in range(-max_step,max_step+1):
            warped=np.roll(obs[0],(dy,dx),axis=(0,1))
            valid=mask[1] & np.roll(mask[0],(dy,dx),axis=(0,1))
            mse=float(np.mean((warped[valid]-obs[1][valid])**2))
            if mse<best:best=mse;best_vel=(dy,dx)
    return best_vel

def motion_ema_with_reset(obs, mask, velocity, reset_times, alpha=.30):
    """Same causal reset times as the proposed lens, fixed EMA gains."""
    out=np.zeros_like(obs)
    state=np.zeros_like(obs[0])
    for t in range(T):
        if t:
            state=phase_shift(state,*velocity)
        if t==0 or t in reset_times:
            state=np.where(mask[t],obs[t],0.)
        else:
            state=np.where(mask[t],(1-alpha)*state+alpha*obs[t],state)
        out[t]=state
    return out


def motion_counted_average(obs,mask,velocity,reset_times):
    """Classical, aligned per-pixel averaging with reset and no process noise."""
    out=np.zeros_like(obs)
    sumw=np.zeros_like(obs[0]); count=np.zeros_like(obs[0])
    for t in range(T):
        if t:
            sumw=np.roll(sumw,velocity,axis=(0,1))
            count=np.roll(count,velocity,axis=(0,1))
        if t in reset_times:
            sumw*=0;count*=0
        sumw+=np.where(mask[t],obs[t],0)
        count+=mask[t]
        out[t]=np.divide(sumw,count,out=np.zeros_like(sumw),where=count>0)
    return out

def spatial_denoise(obs,mask,blur=1.0):
    # Normalized convolution compensates for missing pixels.
    numerator=gaussian_filter(obs,blur,axes=(1,2),mode='wrap')
    denom=gaussian_filter(mask.astype(float),blur,axes=(1,2),mode='wrap')
    return numerator/np.maximum(denom,1e-8)


def score(estimate, clean, mask):
    err=(estimate-clean)**2
    groups={'all':np.ones_like(mask,dtype=bool),
            'visible':mask,'occluded':~mask,
            'pre_occlusion':np.broadcast_to(np.arange(T)[:,None,None]<7,mask.shape),
            'during_occlusion':np.broadcast_to(((np.arange(T)>=7)&(np.arange(T)<=10))[:,None,None],mask.shape),
            'after_change':np.broadcast_to(((np.arange(T)>=13)&(np.arange(T)<=16))[:,None,None],mask.shape)}
    return {k:float(np.mean(err[v])) if np.any(v) else float('nan') for k,v in groups.items()}


def evaluate(seeds=range(200,240),noise=.55, q=.005, reset_threshold=1.7,
             alpha=.30, blur=1.0, switching=True):
    rows=[]; reset_counts=[]; velocity_matches=[]
    for seed in seeds:
        clean,obs,mask,vel=make_episode(int(seed),sigma=noise,switching=switching)
        estimated_vel=infer_velocity_from_first_two(obs,mask)
        velocity_matches.append(estimated_vel==vel)
        phase = temporal_uncertain(obs,mask,vel,sigma=noise,q=q,reset_threshold=reset_threshold,return_diagnostics=True)
        reset_counts.append(len(phase[1]['reset_times']))
        methods={
            'raw':obs,
            'spatial_blur':spatial_denoise(obs,mask,blur=blur),
            'unwarped_ema':temporal_fixed(obs,mask,vel,alpha=alpha,motion=False),
            'motion_ema':temporal_fixed(obs,mask,vel,alpha=alpha,motion=True),
            'motion_ema_same_reset':motion_ema_with_reset(obs,mask,vel,phase[1]['reset_times'],alpha=alpha),
            'motion_counted_average':motion_counted_average(obs,mask,vel,phase[1]['reset_times']),
            'phase_copy_inferred_motion':temporal_uncertain(obs,mask,estimated_vel,sigma=noise,q=q,reset_threshold=reset_threshold),
            'phase_copy_no_reset':temporal_uncertain(obs,mask,vel,sigma=noise,q=q,reset_threshold=None),
            'phase_copy_reset':phase[0],
            'phase_copy_wrong_motion':temporal_uncertain(obs,mask,(0,0),sigma=noise,q=q,reset_threshold=reset_threshold),
        }
        for name,est in methods.items():
            rows.append({'seed':seed,'method':name,**score(est,clean,mask)})
    summary={}
    for name in dict.fromkeys(r['method'] for r in rows):
        z=[r for r in rows if r['method']==name]
        summary[name]={k:{'mse':float(np.nanmean([r[k] for r in z])),
                           'sd':float(np.nanstd([r[k] for r in z],ddof=1))}
                       for k in ('all','visible','occluded','pre_occlusion','during_occlusion','after_change')}
    return rows,summary,{'episodes':len(reset_counts),'mean_resets':float(np.mean(reset_counts)),
                         'episodes_with_resets':int(sum(x>0 for x in reset_counts)),
                         'estimated_velocity_accuracy':float(np.mean(velocity_matches))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--count',type=int,default=40)
    ap.add_argument('--test-start',type=int,default=200)
    ap.add_argument('--noise',type=float,default=.55)
    ap.add_argument('--q',type=float,default=.005)
    ap.add_argument('--reset-threshold',type=float,default=1.7)
    ap.add_argument('--alpha',type=float,default=.30)
    ap.add_argument('--blur',type=float,default=1.0)
    ap.add_argument('--no-switch',action='store_true')
    ap.add_argument('--output',default='results/temporal')
    a=ap.parse_args()
    rows,summ,info=evaluate(range(a.test_start,a.test_start+a.count),noise=a.noise,q=a.q,reset_threshold=a.reset_threshold,alpha=a.alpha,blur=a.blur,switching=not a.no_switch)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with (path.with_suffix('.csv')).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    blob={'configuration':vars(a),'summary':summ,'diagnostics':info}
    path.with_suffix('.json').write_text(json.dumps(blob,indent=2)+'\n')
    print('Mean MSE (smaller is better); all conditions include noisy video and moving occlusion')
    for name,metrics in summ.items():
        print(f"{name:25} all={metrics['all']['mse']:.4f} occluded={metrics['occluded']['mse']:.4f} after_change={metrics['after_change']['mse']:.4f}")
    print(info)

if __name__=='__main__':main()
