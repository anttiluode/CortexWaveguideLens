"""Basic mathematical and causal checks for the phase lens."""
import unittest
import numpy as np
import torch
from waveguide_lens import N, BANDS, SHIFTS, generate, phase_features, AdaptiveLens, run


class LensTests(unittest.TestCase):
    def test_seed_and_dimensions(self):
        a = generate(12, 20, SHIFTS[:32], 0.4)
        b = generate(12, 20, SHIFTS[:32], 0.4)
        self.assertEqual(a['ref'].shape, (12,64))
        self.assertEqual(a['probe'].shape, (12,6))
        np.testing.assert_array_equal(a['query'], b['query'])
        np.testing.assert_array_equal(a['labels'], b['labels'])
        self.assertTrue(np.isfinite(a['probe']).all())

    def test_perfect_phase_alignment_for_shifted_positive(self):
        e = generate(100, 19, SHIFTS, noise_scale=0)
        c,m = phase_features(e)
        # A positive's true shift produces unit coherence in all six bands.
        positives = e['labels']==1
        idx = np.arange(len(e['labels']))[positives]
        np.testing.assert_allclose(c[idx,e['shift'][positives]],1,atol=3e-6)
        self.assertEqual(c.shape,(100,N,len(BANDS)))

    def test_fourier_equivalence_to_classical_phase_correlation(self):
        e=generate(13,21,SHIFTS[:32])
        c,_=phase_features(e)
        a=np.fft.rfft(e['ref'],axis=1)[:,BANDS]
        b=np.fft.rfft(e['query'],axis=1)[:,BANDS]
        norm=b*a.conj()/(np.abs(a)*np.abs(b)+1e-8)
        spectrum=np.zeros((len(a),N),dtype=np.complex128)
        spectrum[:,BANDS]=norm
        # N*inverse-DFT gives sum_b exp(+i2pi k shift/N)*norm.
        spatial=(N*np.fft.ifft(spectrum,axis=1)).real
        np.testing.assert_allclose(spatial,c.sum(-1),rtol=1e-5,atol=1e-5)

    def test_apical_context_has_a_gradient_to_router(self):
        torch.manual_seed(1)
        model=AdaptiveLens(True)
        e=generate(24,8,SHIFTS[:32])
        c,_=phase_features(e)
        output=model(torch.from_numpy(c),torch.from_numpy(e['probe']))
        output.square().mean().backward()
        self.assertGreater(model.router[0].weight.grad.abs().sum().item(),0.01)

    def test_small_training_can_use_context(self):
        # Fixed, deterministic smoke test; full numerical results use 3 seeds.
        r=run(seed=3,steps=130,train_size=800,test_size=500)
        self.assertGreater(r['apical_probe']['accuracy'],r['fixed_gain']['accuracy']+0.03)
        self.assertGreater(r['apical_probe']['accuracy'],r['shuffled_apical_probe']['accuracy']+0.03)
        self.assertGreater(r['inverse_noise_classical']['accuracy'],0.85)

if __name__ == '__main__':
    unittest.main()
