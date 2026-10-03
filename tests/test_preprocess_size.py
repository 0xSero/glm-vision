import unittest

from build_art_dataset import preprocess_size


class PreprocessSizeTests(unittest.TestCase):
    def test_short_side_rounded_up_stays_within_the_pixel_cap(self):
        width, height = preprocess_size(12000, 20, 300_000)
        self.assertEqual(width % 28, 0)
        self.assertEqual(height % 28, 0)
        self.assertGreaterEqual(width, 28)
        self.assertGreaterEqual(height, 28)
        self.assertLessEqual(width * height, 300_000)

    def test_sizes_already_under_the_cap_keep_their_grid(self):
        self.assertEqual(preprocess_size(224, 224, 300_000), (224, 224))
        self.assertEqual(preprocess_size(384, 512, 300_000), (364, 504))


if __name__ == "__main__":
    unittest.main()
