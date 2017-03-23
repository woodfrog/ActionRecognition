import unittest
import numpy as np
from OF_utils import stack_optical_flow


class TestOpticalFlow(unittest.TestCase):
    def test_stacked_OF(self):
        frames = np.ndarray((20, 28, 28, 3))
        flows = stack_optical_flow(frames, mean_sub=True)
        self.assertEqual(flows.shape, (28, 28, 38))
