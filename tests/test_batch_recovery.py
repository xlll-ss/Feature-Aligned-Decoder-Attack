import unittest

import torch

from attack_batch_candidates import candidate_rows, perturb_observation, relative_error


class BatchRecoveryTest(unittest.TestCase):
    def test_unique_high_confidence_rows_approximate_features(self):
        features = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]])
        labels = torch.tensor([0, 1])
        probabilities = torch.full((2, 100), 0.2 / 99)
        probabilities[0, 0] = 0.8
        probabilities[1, 1] = 0.8
        residuals = probabilities - torch.nn.functional.one_hot(labels, num_classes=100).float()
        grad_w = residuals.t() @ features / len(features)
        grad_b = residuals.mean(dim=0)
        rows, _ = candidate_rows(grad_b, absolute_threshold=1e-10, relative_threshold=1e-4)
        self.assertEqual(rows, [0, 1])
        recovered_zero = (grad_w[0] / grad_b[0]).view(1, -1)
        recovered_one = (grad_w[1] / grad_b[1]).view(1, -1)
        self.assertLess(relative_error(recovered_zero, features[0:1]), 0.02)
        self.assertLess(relative_error(recovered_one, features[1:2]), 0.02)

    def test_observation_clip_and_quantization(self):
        grad_w = torch.tensor([[3.0, 4.0], [-1.0, 0.0]])
        grad_b = torch.tensor([2.0, -2.0])
        clipped_w, clipped_b = perturb_observation(
            grad_w,
            grad_b,
            clip_norm=1.0,
            quantization_bits=8,
        )
        norm = torch.sqrt(torch.sum(clipped_w.square()) + torch.sum(clipped_b.square()))
        self.assertLessEqual(float(norm), 1.00001)
        self.assertTrue(torch.isfinite(clipped_w).all())
        self.assertTrue(torch.isfinite(clipped_b).all())


if __name__ == "__main__":
    unittest.main()
