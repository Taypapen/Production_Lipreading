import unittest
import cv2
import numpy as np
from video_preprocess import VideoPreprocessor


class VideoPreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = VideoPreprocessor()
        self.no_face_image = cv2.imread('/home/taylorpap/Downloads/corn-field.jpeg')
        self.face_image = cv2.imread('/home/taylorpap/Downloads/person-2659138773.jpeg')
        self.non_square_array = np.asarray(self.no_face_image)
        self.square_array = np.asarray(self.face_image)
        self.missing_landmarks_video = '/media/taylorpap/1TBM2/DatasetML/lipread_mp4/INFORMATION/train/INFORMATION_00540.mp4'
        self.known_good_video = '/media/taylorpap/1TBM2/DatasetML/lipread_mp4/INFORMATION/train/INFORMATION_00541.mp4'
        self.non_standard_video = '/home/taylorpap/Bootcamp/IMG_1922.mov'

    def test_frame_detection(self):
        self.assertIsNone(self.preprocessor.detect_Facial_Landmarks(cv2.cvtColor(self.no_face_image, cv2.COLOR_BGR2RGB)))
        self.assertIsInstance(self.preprocessor.detect_Facial_Landmarks(cv2.cvtColor(self.face_image, cv2.COLOR_BGR2RGB)), dict)

    def test_avg_face(self):
        # Make sure face oval averages loaded
        self.assertIsNotNone(self.preprocessor.face_oval_avgs)

    def test_square_cropping(self):
        square_image = self.preprocessor.square_crop(self.non_square_array,
                                                     self.non_square_array.shape[0], self.non_square_array.shape[1])
        self.assertEqual(square_image.shape[0], square_image.shape[1])
        already_square_image = self.preprocessor.square_crop(self.square_array,
                                                     self.square_array.shape[0], self.square_array.shape[1])
        self.assertEqual(already_square_image.shape[0], already_square_image.shape[1])
        # Use wrong height/width so output is not square (Edge Case)
        with self.assertRaises(AssertionError):
            self.preprocessor.square_crop(self.non_square_array, 800, 1000)

    def test_get_face_points(self):
        self.assertFalse(self.preprocessor.get_face_points(self.missing_landmarks_video))
        grey_output = self.preprocessor.get_face_points(self.known_good_video)
        self.assertEqual(grey_output.shape[1], 96)
        self.assertEqual(grey_output.shape[2], 96)
        non_standard_grey_output = self.preprocessor.get_face_points(self.non_standard_video)
        self.assertEqual(non_standard_grey_output.shape[1], 96)
        self.assertEqual(non_standard_grey_output.shape[2], 96)


if __name__ == '__main__':
    unittest.main()
