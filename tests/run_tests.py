import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'important_code'))

if __name__ == '__main__':
    suite = unittest.TestLoader().discover(str(ROOT / 'tests'))
    suite.addTests(unittest.TestLoader().discover(
        str(ROOT / 'important_code'), pattern='test_target_checkpoint_layout.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
