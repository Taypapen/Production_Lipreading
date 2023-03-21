# Production_Lipreading

## LIPREADING APP
Deployed Streamlit app can be accessed at: https://single-word-lipreading.streamlit.app/

## Repository Functions

This Repo is composed of 3 different primary functions:

1. Streamlit App for Reading Lips from single word videos
2. Script to preprocess LRW (Lip Reading in the Wild) Dataset
3. Script to train custom Pytorch model

## Set-Up:

Clone the Directory

Create a new Conda Environment using *environment.yml*


`conda env create -n ENVNAME --file environment.yml`

If training a model:
Download LRW Dataset (https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

If having issues with ffmpeg, download from ffmpeg.org/download and place executable ffmpeg and ffprobe in main directory. (Allows command line version of app to run on MacOS)

## Lip Reading App

To run app, activate created conda environment, cd to the correct directory and use command:

`streamlit run video_prediction_app.py`

This will open a browser window with the app. From there, upload a video of your choice to get the Model Prediction.
For best results: Mouth should be centered in frame of video and video should be trimmed to just before/after the word is spoken (ok to have part of words spoken before/after target). Only works on single word videos.

## Pre-processing Dataset

To Preprocess LRW videos:

`python Crop_and_Preprocess_Videos.py --data-direc <SAVE_LOCATION> --lrw-direc <LRW_DIRECTORY>`

(LRW_DIREC should be path of Directory which contains word folders inside)
Saved preprocessed and cropped arrays will be saved in SAVE_LOCATION to be used for training. Directory Structure is same as LRW Dataset.

## Model Training

(CUDA DEVICES ONLY)
For starting training using the pretrained Lipread3 model:
  
```
python ./main.py \
--model-path  ./models/pretrained_weights/Lipread3/model_weights.tar \
--data-direc <PREPROCESSED_VIDEO_DIREC> \
--lrw-direc <LRW_DIREC>
```

Default options is --epochs: 80, --lr: 0.001, --batch-size: 64, --workers: 8
These can all be specified to different values if desired.
  
A checkpoint is saved after each Epoch during training. To resume training from a checkpoint:
  
```
python ./main.py \
--model-path  <CHECKPOINT_PATH> \
--data-direc <PREPROCESSED_VIDEO_DIREC> \
--lrw-direc <LRW_DIREC> \
--model-weights-only False
```

If using partial dataset: use --wordlist-file to point to correct list.txt of words, and change --num-classes to match correct number of words

Sample of some randomly chosen words from dataset can be found at https://drive.google.com/drive/folders/1ovInsxuZOsub6-Hw0rQkB_PfN2qBUNgp?usp=share_link
Wordlist for sample data is: sample_wordlist.txt
