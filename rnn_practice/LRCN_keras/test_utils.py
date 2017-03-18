import unittest
import os
from LRCN_utils import get_data_list, video_image_generator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        # test on 2 directories
        data_dir = '/Users/cjc/cv/ActionRecognition_rnn/data/data'
        list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
        video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

        self.train_data, self.test_data, _ = get_data_list(list_dir, video_dir)
        self.assertEqual(len(self.train_data), 183)
        self.assertEqual(len(self.test_data), 76)

    def test_image_generator(self):
        image_gen = video_image_generator(self.train_data, 20, 10, (216, 216), 101)
        batch_image, batch_label = next(image_gen)
        self.assertEqual(batch_image.shape, (20, 216, 216, 3))
        self.assertEqual(batch_label.shape, (20, 101))


if __name__ == '__main__':
    unittest.main()
