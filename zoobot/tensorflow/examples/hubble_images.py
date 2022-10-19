import os
import logging
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tqdm import tqdm

from zoobot.shared import label_metadata, schemas
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess, custom_layers
from zoobot.tensorflow.predictions import predict_on_dataset
from zoobot.tensorflow.training import training_config, losses
from zoobot.tensorflow.transfer_learning import utils

#Image parameters
initial_size = 300 
crop_size = int(initial_size * 0.75)
resize_size = 224  #Zoobot, as pretrained, expects 224x224 images
file_format = "png"

#Batch size
batch_size = 128

#Read in the dataset
df = pd.read_csv("/content/drive/MyDrive/MPE/2022_Ben_Aussel/Data/Hubble_COSMOS_labels_complete.csv")

