# Production_Lipreading

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

## Lip Reading App

To run app, activate created conda environment, cd to the correct directory and use command:

`streamlit run video_prediction_app.py`

This will open a browser window with the app. From there, upload a video of your choice to get the Model Prediction.
For best results: Mouth should be centered in frame of video and video should be trimmed to just before/after the word is spoken. Only works on single word videos.

## Pre-processing Dataset

To Preprocess LRW videos:

`python lipreading/Crop_and_Preprocess_Videos.py --data-direc <SAVE_LOCATION> --lrw-direc <LRW_BASEDIRECTORY>`

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

