import cv2
import itertools
import numpy as np
import time
import mediapipe as mp
import re
from lipreading.preprocesses import *
import torch
import os
from skimage import transform as transf
from tqdm import tqdm


class VideoPreprocessor(object):
    def __init__(self, avg_face_pth= './lipreading/face_oval_averages.npz'):

        self.init_crop_width = 96
        self.init_crop_height = 96
        self.actual_crop_size = (88, 88)
        self.fps = 25
        self.frames_adjust = True

        self.landmark_crop_indexes = [2, 3, 10, 11, 26, 30]
        self.std_size = (256, 256)
        self.set_up_avg_face(avg_face_pth)

    def set_up_avg_face(self, avg_face_pth):
        assert os.path.isfile(avg_face_pth), "Face avg path: {} does not exist. Specify valid path".format(avg_face_pth)
        face_oval_avgs = np.load(avg_face_pth, allow_pickle=True)['data']
        self.face_oval_avgs = face_oval_avgs * 256

    def extract_points_from_mesh(self, face_landmarks, indexes):
        points_data_regex = re.compile(r'\d\.\d+')
        xy_points_list = []
        for count, each_index in enumerate(indexes):
            xyzpointsraw = face_landmarks.landmark[each_index]
            points_list = points_data_regex.findall(str(xyzpointsraw))
            if len(points_list) < 1:
                xy_points_list.append([None])
            else:
                xyclean = [float(points_list[0]), float(points_list[1])]
                xy_points_list.append(xyclean)
        xy_points_array = np.array(xy_points_list)
        return xy_points_array

    def detect_Facial_Landmarks(self, image):
        face_mesh_videos, mp_face_mesh = self.mediapipe_setup()
        frame_dict = {}
        mesh_result = face_mesh_videos.process(image)
        oval_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
        lips_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
        if mesh_result.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(mesh_result.multi_face_landmarks):
                oval_points_array = self.extract_points_from_mesh(face_landmarks, oval_indexes)
                lips_points_array = self.extract_points_from_mesh(face_landmarks, lips_indexes)
                frame_dict['oval_landmarks'] = oval_points_array
                frame_dict['lips_landmarks'] = lips_points_array
        else:
            frame_dict = None
        return frame_dict

    def mediapipe_setup(self):
        # initialize the mediapipe face mesh class
        mp_face_mesh = mp.solutions.face_mesh
        # Setup the face landmarks function for videos
        face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                 min_detection_confidence=0.5, min_tracking_confidence=0.3)
        return face_mesh_videos, mp_face_mesh

    def get_face_points(self, video, output_path=None):
        assert os.path.isfile(video), "Video: {} Not found".format(video)
        vid_capture = cv2.VideoCapture(video)
        frame_idx = 0
        sequence = []
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
            return False
        else:
            self.fps = int(vid_capture.get(5))
            if output_path:
                output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps,
                                               (self.init_crop_width,self.init_crop_height))
            while (vid_capture.isOpened()):
                width = int(vid_capture.get(3))
                height = int(vid_capture.get(4))
                ret, frame = vid_capture.read()
                if ret:
                    if width != height:
                        frame = self.square_crop(frame, height, width)
                    frame_points_dict = self.detect_Facial_Landmarks(frame)
                    # Abort if unable to detect Landmarks
                    if frame_points_dict is None:
                        print("Unable to Detect Facial Landmarks in frame {} of video, please select another video"
                              .format(str(frame_idx)))
                        vid_capture.release()
                        if output_path:
                            output_video.release()
                        cv2.destroyAllWindows()
                        return False
                    if frame.shape[0] != 256:
                        frame = cv2.resize(frame, self.std_size, interpolation=cv2.INTER_LINEAR)
                    current_oval = frame_points_dict['oval_landmarks'] * 256
                    if frame_idx == 0:
                        transformed_frame, trans_mat = self.warp_img(current_oval[self.landmark_crop_indexes, :],
                                                                    self.face_oval_avgs[self.landmark_crop_indexes, :],
                                                                    frame,
                                                                    self.std_size)
                    current_lips = frame_points_dict['lips_landmarks']
                    trans_lips = trans_mat(current_lips * 256)
                    trans_frame = self.apply_transform(trans_mat, frame, self.std_size)
                    cut_frame = self.crop_out_patch(trans_frame, trans_lips, self.init_crop_height // 2, self.init_crop_width // 2)
                    if output_path:
                        output_video.write(cut_frame)
                    sequence.append(cut_frame)
                    frame_idx += 1
                else:
                    break
        vid_capture.release()
        if output_path:
            output_video.release()
        cv2.destroyAllWindows()
        if self.frames_adjust:
            if len(sequence) > 29:
                sequence = self.adjust_frames(sequence)
        return self.convert_bgr2gray(np.array(sequence))

    # Warp image and get transform parameters
    def warp_img(self, src, dst, img, std_size):
        tform = transf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = transf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    # Apply a previously calculated transform
    def apply_transform(self, transform, img, std_size):
        warped = transf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped

    def crop_out_patch(self, img, landmarks, height, width):
        center_x, center_y = np.mean(landmarks, axis=0)

        cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                             int(round(center_x) - round(width)): int(round(center_x) + round(width))])
        return cutted_img

    def square_crop(self, img, height, width):
        diff_val = int((height - width)//2)
        if diff_val < 0:
            cut_frame = np.copy(img[:, abs(diff_val): (width - abs(diff_val))])
        elif diff_val > 0:
            cut_frame = np.copy(img[diff_val: (height-diff_val), :])
        else:
            return img
        # Check to make sure frame size is roughly equal (In case of odd resolutions)
        assert abs(cut_frame.shape[0] - cut_frame.shape[1]) <= 1, "Unable to Crop to 1:1 aspect ratio"
        return cut_frame

    def adjust_frames(self, sequence):
        if 39 < self.fps <= 60:
            sequence = sequence[::2]
        elif 60 < self.fps <= 90:
            sequence = sequence[::3]
        elif 90 < self.fps <= 120:
            sequence = sequence[::4]
        half_point = int((len(sequence)-1)//2)
        return sequence

    def convert_bgr2gray(self, data):
        return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)

    def preprocess_creation(self):
        crop_size = self.actual_crop_size
        (mean, std) = (0.421, 0.165)
        preprocessed = Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std)])
        return preprocessed

    def tensor_setup_from_array(self, video_array):
        preprocess_video = self.preprocess_creation()
        video_tensor = torch.Tensor(video_array)
        preprocessed_tensor = preprocess_video(video_tensor)
        return preprocessed_tensor.unsqueeze(0), len(video_array)

    def full_tensor_setup(self, video):
        video_array = self.get_face_points(video)
        preprocess_video = self.preprocess_creation()
        video_tensor = torch.Tensor(video_array)
        preprocessed_tensor = preprocess_video(video_tensor)
        return preprocessed_tensor.unsqueeze(0), len(video_array)

