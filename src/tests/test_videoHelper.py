from unittest import TestCase

from src.data import DataSet
from src.utils import VideoHelper


class TestVideoHelper(TestCase):

    # def setUp(self):
    #     self.video_helper = VideoHelper()

    def test_clear_part_video_filename(self):
        expected = 'video45_2'
        actual = VideoHelper._clear_video_filename('video45_2')
        self.assertEqual(expected, actual)

    def test_clear_usual_video_filename(self):
        expected = '79-30-960x720'
        actual = VideoHelper._clear_video_filename('79-30-960x720')
        self.assertEqual(expected, actual)

    def test_clear_right_subject_video_filename(self):
        expected = 'video59'
        actual = VideoHelper._clear_video_filename('video59_right')
        self.assertEqual(expected, actual)

    def test_get_num_frames(self):
        video_filename = "79-30-960x720"
        expected = len(DataSet.get_targets('test', video_filename))
        actual = VideoHelper._extract_num_frames(video_filename)
        self.assertEqual(expected, actual)