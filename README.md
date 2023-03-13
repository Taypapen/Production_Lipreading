# Production_Lipreading

## Repository Functions

This Repo is composed of 3 different primary functions:

1. Streamlit App for Reading Lips from single word videos
2. Script to preprocess LRW (Lip Reading in the Wild) Dataset
3. Script to train custom Pytorch model

## Set-Up:

Clone the Directory

Create a new Conda Environment using *environment.yml*
> conda env create -n ENVNAME --file environment.yml

If training a model:
Download LRW Dataset (https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

## Lip Reading App

To run app, activate created conda environment, cd to the correct directory and use command:
> streamlit run video_prediction_app.py

This will open a browser window with the app. From there, upload a video of your choice to get the Model Prediction.
For best results: Mouth should be centered in frame of video and video should be trimmed to just before/after the word is spoken. Only works on single word videos.

## Pre-processing Dataset

To Preprocess LRW videos:
> lipreading/Crop_and_Preprocess_Videos.py --data-direc <SAVELOCATION> --lrw-direc <LRW_BASEDIRECTORY>

Saved preprocessed and cropped arrays will be saved in <SAVELOCATION> to be used for training. Directory Structure is same as LRW Dataset.

