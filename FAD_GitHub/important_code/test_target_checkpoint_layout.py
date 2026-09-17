import sys
import unittest
from pathlib import Path


IMPORTANT_CODE = Path(__file__).resolve().parents[1] / "important_code"
sys.path.insert(0, str(IMPORTANT_CODE))

from train_decoder_multidataset import remap_vgg_feature_key


class TargetCheckpointLayoutTest(unittest.TestCase):
    def test_remaps_official_undefended_vgg_blocks(self):
        cases = {
            "feature.0.weight": "layer1.0.weight",
            "feature.4.running_var": "layer1.4.running_var",
            "feature.7.weight": "layer2.0.weight",
            "feature.14.bias": "layer3.0.bias",
            "feature.24.weight": "layer4.0.weight",
            "feature.41.num_batches_tracked": "layer5.7.num_batches_tracked",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(remap_vgg_feature_key(source), expected)

    def test_keeps_bido_and_non_feature_keys_unchanged(self):
        for key in ("layer1.0.weight", "bn.weight", "fc_layer.bias", "feature.bad.weight"):
            with self.subTest(key=key):
                self.assertEqual(remap_vgg_feature_key(key), key)


if __name__ == "__main__":
    unittest.main()
