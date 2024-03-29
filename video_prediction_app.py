import os
import streamlit as st
import subprocess as sp
from models.pytorch_nn import Lipread3
from lipreading.video_preprocess import VideoPreprocessor
import torch.nn.functional as F
import shutil
import logging

import torch

@st.cache_data
def logging_setup():
    logging.basicConfig(filename='app_log.log', level=logging.INFO)
    logging.info("Logger set up")

st.title("Lipreading Single Word")

uploaded_video = st.file_uploader("Choose a video File...")

def get_wordslist_from_txt_file(file_path):
    assert os.path.isfile(file_path), "Word List file does not exist. Path input: {}".format(
        file_path)
    with open(file_path) as file:
        word_list = file.readlines()
        word_list = [item.rstrip() for item in word_list]
    return word_list

def convert_mp4v_to_h264(infile,outfile):
    """
    Uses ffmpeg executable/binary file present in directory, otherwise tries to run installed ffmpeg.
    """
    if os.path.isfile("./ffmpeg"):
        sp.call(args=f"./ffmpeg -y -i {infile} -c:v libx264 {outfile}".split(" "))
    else:
        sp.call(args=f"ffmpeg -y -i {infile} -c:v libx264 {outfile}".split(" "))
    os.remove(infile)

def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet.
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

logging_setup()

#Clear saved videos from previous prediction
if not uploaded_video and os.path.isdir('./tmp'):
    shutil.rmtree('./tmp')

logging.info("Creating paths...")

os.makedirs('./tmp', exist_ok=True)
wordslist_file = './wordlist.txt'
model_path = './models/pretrained_weights/Lipread3/model_weights.tar'
opened_upload = './tmp/uploaded_video_bytes'
converted_path = './tmp/h264video.mp4'
cropped_video_mp4v = './tmp/cropped_mp4v.mp4'
cropped_video = './tmp/cropped_h264.mp4'

logging.info("Paths Created")


@st.cache_resource
def load_model():

    model = Lipread3(500)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    loaded_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(loaded_state_dict, strict=False)
    model.eval()
    logging.info("Model Loaded")
    # Load words_list for referencing answer from prediction
    words_list = get_wordslist_from_txt_file(wordslist_file)
    logging.info("Words list Loaded")
    return model, words_list


@st.cache_data
def main(uploaded_video, opened_upload):

    video_load_state = st.text('Loading Video and Preprocessing...')
    # Load Video and set up as tensor for input
    video_setup = VideoPreprocessor()
    logging.info("Cropping Video for visual...")
    cropped_array = video_setup.get_face_points(opened_upload, output_path=cropped_video_mp4v)
    video_tensor, video_length = video_setup.tensor_setup_from_array(cropped_array)
    logging.info("Converting Videos to h264...")
    convert_mp4v_to_h264(opened_upload, converted_path)
    convert_mp4v_to_h264(cropped_video_mp4v, cropped_video)
    logging.info("Videos converted")
    video_load_state.text('Video Ready!')
    # Input video data into model and print word prediction
    prediction = model(video_tensor, lengths=[video_length])
    return prediction

if uploaded_video is not None:
    logging.info("Loading Lipread3 model and getting Words list")
    model, words_list = load_model()
    logging.info("Creating video file from uploaded data...")
    write_bytesio_to_file(opened_upload, uploaded_video)
    prediction = main(uploaded_video, opened_upload)
    logging.info("Prediction obtained from model")
    col1, col2 = st.columns(2)
    col1.header("Input Video:")
    col1.video(converted_path)
    col2.header("Transformed Video:")
    col2.video(cropped_video)
    confidences, guesses = torch.topk(F.softmax(prediction, dim=1).data, 2, dim=1)
    st.header("THE WORD IS: " + words_list[guesses[0][0]])
    st.write("Confidence score of: " + str(round(confidences[0][0].item() * 100, 2)))
    second = words_list[guesses[0][1]]
    second_soft = round(confidences[0][1].item() * 100, 2)
    if st.checkbox('Show Second Guess', value=False):
        st.write("Second Guess is: " + second)
        st.write("Confidence Score of: " + str(second_soft))

st.write("Check Out the Source Code at https://github.com/Taypapen/Production_Lipreading/")
