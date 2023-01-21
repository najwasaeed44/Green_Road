# import os
# import cv2
# import glob
# import random
import numpy as np
# import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
from keras.optimizers import Adam
import plotly.graph_objects as go
from keras.layers import LeakyReLU
# from plotly.subplots import make_subplots
from helper import potholes_predictions, DenseNet_UNET, DataGenerator, loss_function, accuracy_function, get_df, plot_3d_prediction



if __name__ == "__main__":
  folder_path = '/content/testing/rgb' + '/*'
  pothole_classifier_model_path = '/content/drive/MyDrive/smartathon/potholes_detection_model.sav'
  depth_model_estimation_path = '/content/drive/MyDrive/smartathon/GRAY_DEPTH-MODEL-WITH-nyu-depth-ex1.h5'
  mask_labelling_model_path = '/content/drive/MyDrive/smartathon/Label-MODEL-WITH-agumentation-ex1.h5'

  print('Loading the models...\n\n\n')
  mask_model = DenseNet_UNET(64, LeakyReLU(0.2))
  mask_model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.BinaryIoU(), 
                    tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

  mask_model.load_weights(mask_labelling_model_path)


  depth_model = tf.keras.models.load_model(depth_model_estimation_path, custom_objects={'loss_function':loss_function, 'accuracy_function':accuracy_function})

  print('Models weights have been loaded successfully 😊😊😊\n\n\n')
  print('Running the pothole(s) classifier model...\n\n\n')
  df = potholes_predictions(folder_path, pothole_classifier_model_path)

  potholes_generator = DataGenerator(df, 'mask', batch_size=df.shape[0], shuffle=False, dim=(320,320))
  potholes_images, _ = next(iter(potholes_generator))
  print('\n\n\n Running the mask(s) model...\n\n\n')
  mask_potholes_preds = mask_model.predict(np.array(potholes_images))
  print('\n\n\n Running the Depth estimation model...\n\n\n')
  depth_potholes_preds = depth_model.predict(np.array(potholes_images))

  print("\n\n\n Let's build the 3D scene for the first 4 potholes 😊😊😊 \n\n\n")

  for i in range(4):
    plot_3d_prediction(depth_potholes_preds[i], mask_potholes_preds[i])