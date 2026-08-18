import unittest

import torch
import tqdm

class MyTestCase(unittest.TestCase):
    def test_cuda_is_available(self):
        self.assertEqual(torch.cuda.is_available(), True)  # add assertion here

    def test_tqdm_has_versio(self):
        self.assertIn("4.68.2", tqdm.__version__)

if __name__ == '__main__':
    unittest.main()
