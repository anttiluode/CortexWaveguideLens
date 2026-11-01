"""Invariant and causal tests for the stage-2 visual signal-copy experiment."""
import unittest
import numpy as np
from temporal_wavecopy import (N, T, phase_shift, image_scene, make_episode,
    infer_velocity_from_first_two, temporal_uncertain, motion_counted_average)

class TemporalWavecopyTests(unittest.TestCase):
    def test_fourier_shift_matches_periodic_translation(self):
        rng=np.random.default_rng(12);img=rng.normal(size=(N,N))
        np.testing.assert_allclose(phase_shift(img,2,-1),np.roll(img,(2,-1),axis=(0,1)),atol=1e-12)

    def test_motion_estimation_uses_only_first_two_frames(self):
        clean,obs,mask,vel=make_episode(204)
        original=infer_velocity_from_first_two(obs,mask)
        modified=obs.copy();modified[2:]=12345
        self.assertEqual(original,infer_velocity_from_first_two(modified,mask))
        self.assertEqual(original,vel)

    def test_no_future_observation_affects_past_output(self):
        _,obs,mask,vel=make_episode(205)
        first=temporal_uncertain(obs,mask,vel,q=0,reset_threshold=1.3)
        modified=obs.copy(); modified[10:]+=10
        second=temporal_uncertain(modified,mask,vel,q=0,reset_threshold=1.3)
        np.testing.assert_array_equal(first[:10],second[:10])

    def test_zero_process_noise_is_classical_counted_average(self):
        _,obs,mask,vel=make_episode(206)
        est,diag=temporal_uncertain(obs,mask,vel,q=0,reset_threshold=1.3,return_diagnostics=True)
        average=motion_counted_average(obs,mask,vel,diag['reset_times'])
        np.testing.assert_allclose(est,average,rtol=1e-10,atol=1e-10)

    def test_scene_switch_resets_not_random_frames(self):
        _,obs,mask,vel=make_episode(207,switching=True)
        _,diag=temporal_uncertain(obs,mask,vel,q=0,reset_threshold=1.3,return_diagnostics=True)
        self.assertEqual(diag['reset_times'],[13])
        _,obs,mask,vel=make_episode(207,switching=False)
        _,diag=temporal_uncertain(obs,mask,vel,q=0,reset_threshold=1.3,return_diagnostics=True)
        self.assertEqual(diag['reset_times'],[])

    def test_wrong_motion_and_no_reset_cause_errors(self):
        clean,obs,mask,vel=make_episode(208)
        good=temporal_uncertain(obs,mask,vel,q=0,reset_threshold=1.3)
        wrong=temporal_uncertain(obs,mask,(0,0),q=0,reset_threshold=1.3)
        noreset=temporal_uncertain(obs,mask,vel,q=0,reset_threshold=None)
        self.assertLess(np.mean((good-clean)**2),np.mean((wrong-clean)**2))
        self.assertLess(np.mean((good[13:17]-clean[13:17])**2),
                        np.mean((noreset[13:17]-clean[13:17])**2))

if __name__ == '__main__':unittest.main()
