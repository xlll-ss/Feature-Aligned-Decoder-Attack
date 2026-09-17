import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiment_utils import ManifestImageDataset, build_split_manifest, summarize_values


class ExperimentUtilsTest(unittest.TestCase):
    def test_manifest_partitions_are_disjoint_and_labeled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "images"
            root.mkdir()
            labels = []
            for index in range(20):
                name = f"{index:06d}.jpg"
                Image.new("RGB", (8, 8), color=(index, index, index)).save(root / name)
                labels.append(f"{name} {index // 2 + 1}")
            label_file = Path(temp_dir) / "identity.txt"
            label_file.write_text("\n".join(labels), encoding="utf-8")
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest = build_split_manifest(
                root,
                manifest_path,
                seed=7,
                auxiliary_fraction=0.6,
                validation_fraction=0.2,
                label_file=label_file,
                label_offset=-1,
                group_by_label=True,
            )
            self.assertEqual(manifest["labeled_image_count"], 20)
            partition_paths = {
                partition: {path for path, value in manifest["assignments"].items() if value == partition}
                for partition in ("auxiliary", "validation", "test")
            }
            self.assertFalse(partition_paths["auxiliary"] & partition_paths["validation"])
            self.assertFalse(partition_paths["auxiliary"] & partition_paths["test"])
            self.assertFalse(partition_paths["validation"] & partition_paths["test"])
            partition_labels = {
                partition: {manifest["labels"][path] for path in paths}
                for partition, paths in partition_paths.items()
            }
            self.assertFalse(partition_labels["auxiliary"] & partition_labels["validation"])
            self.assertFalse(partition_labels["auxiliary"] & partition_labels["test"])
            self.assertFalse(partition_labels["validation"] & partition_labels["test"])
            test_dataset = ManifestImageDataset(manifest_path, "test")
            self.assertTrue(test_dataset.has_complete_labels)
            _, label = test_dataset[0]
            self.assertGreaterEqual(label, 0)
            json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_summary_contains_bootstrap_interval(self):
        summary = summarize_values([1, 2, 3, 4], seed=9)
        self.assertEqual(summary["n"], 4)
        self.assertLessEqual(summary["ci95"]["lower"], summary["mean"])
        self.assertGreaterEqual(summary["ci95"]["upper"], summary["mean"])


if __name__ == "__main__":
    unittest.main()
