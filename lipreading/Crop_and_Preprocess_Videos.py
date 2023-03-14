import os
import glob
import numpy as np
import argparse
from lipreading.video_preprocess import VideoPreprocessor

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading')
    
    parser.add_argument('--data-direc', default=None, help='Where to save Preprocessed Videos/Data')
    parser.add_argument('--lrw-direc', default=None, help='LRW Data Directory')

    args = parser.parse_args()
    return args

args = load_args()

def save2npz(filename, data=None):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)

def preprocess_extract_save(video, output_loc, preprocessor):
    dir, file = os.path.split(video)
    part, which_folder = os.path.split(dir)
    _, word = os.path.split(part)
    new_filename = file[:-4] + ".npz"
    new_save_output = os.path.join(output_loc, word, which_folder, new_filename)
    if not os.path.exists(new_save_output):
        video_array = preprocessor.get_face_points(video)
        if not video_array:
            return False
        save2npz(new_save_output, video_array)

def main():
    assert os.path.isdir(args.data_direc), "Preprocessed video directory does not exist. Path input {}".format(args.data_direc)
    assert os.path.isdir(args.lrw_direc), "LipReadinginTheWild direc does not exist. Path input {}".format(args.lrw_direc)
    videos = glob.glob(os.path.join(args.lrw_direc, '*', '*', '*.mp4'))
    Preprocessor = VideoPreprocessor()
    for video in videos:
        preprocess_extract_save(video, args.data_direc, Preprocessor)

if __name__ == '__main__':
    main()