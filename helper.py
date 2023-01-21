import os
import cv2
import glob
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
import tensorflow as tf
import scipy.stats as stat
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import skimage.feature as feature
import plotly.graph_objects as go
from keras.models import Model, load_model
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.applications import DenseNet201
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, concatenate, add


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image





@njit
def get_pixel(img, center, x, y):
  new_value = 0
  try:
    if img[x][y] >= center + 5:
      new_value = 1
    elif img[x][y] <= center - 5:
      new_value = -1
    elif np.abs(img[x][y] - center) < 5:
      new_value = 0

  except:
    pass
      
  return new_value



@njit
def ltp_calculated_pixel(img, x, y):

	center = img[x][y]
	val_ar = []
	# top_left
	val_ar.append(get_pixel(img, center, x-1, y-1))
	# top
	val_ar.append(get_pixel(img, center, x-1, y))
	# top_right
	val_ar.append(get_pixel(img, center, x-1, y + 1))
	# right
	val_ar.append(get_pixel(img, center, x, y + 1))
	# bottom_right
	val_ar.append(get_pixel(img, center, x + 1, y + 1))
	# bottom
	val_ar.append(get_pixel(img, center, x + 1, y))
	# bottom_left
	val_ar.append(get_pixel(img, center, x + 1, y-1))
	# left
	val_ar.append(get_pixel(img, center, x, y-1))
	# Now, we need to convert binary
	# values to decimal
	power_val = [1, 2, 4, 8, 16, 32, 64, 128]
	val = [val_ar[i] * power_val[i] for i in np.arange(len(val_ar))]
	return sum(val)
 

@njit

def general_img_info(image, contrast, dissimilarity, homogeneity, energy, correlation, ASM):


  info = [np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2]), # find the RGB mean
                      
                      np.std(image[:, :, 0]), np.std(image[:, :, 1]), np.std(image[:, :, 2]), # find the RGB std

                      contrast[0][0], contrast[0][1], contrast[0][2], contrast[0][3],# find the RGB contrast
                      
                      dissimilarity[0][0], dissimilarity[0][1], dissimilarity[0][2], dissimilarity[0][3],# find the RGB dissimilarity
                      
                      homogeneity[0][0], homogeneity[0][1], homogeneity[0][2], homogeneity[0][3],# find the RGB homogeneity
                      
                      energy[0][0], energy[0][1], energy[0][2], energy[0][3],# find the RGB energy
                      
                      correlation[0][0], correlation[0][1], correlation[0][2], correlation[0][3],# find the RGB correlation
                      
                      ASM[0][0], ASM[0][1], ASM[0][2], ASM[0][3]]# find the RGB ASM
  return info



@njit
def ltp_hist(img_ltp, img_size, img_gray, hist_dim):

  results = [[i, j] for i in np.arange(img_size) for j in np.arange(img_size)]

  for k in np.arange(len(results)):
    img_ltp[results[k][0], results[k][1]] = ltp_calculated_pixel(img_gray, results[k][0], results[k][1])

  hist, _ = np.histogram(img_ltp.ravel(), bins=np.arange(0, hist_dim), range=(0, hist_dim))

  return hist

  
def features_extraction(folder_path, img_size, hist_dim, mask=True):

  features = []
  path = glob.glob(folder_path)
  for i in tqdm(range(len(path))):
    image = cv2.imread(path[i])
    img_ltp = np.zeros((img_size, img_size),np.uint8)

    vertices = np.array([[(0, image.shape[0]), (0, 400), (700, 380), (1250, 380), (image.shape[1],410), (image.shape[1],image.shape[0])]], dtype=np.int32)
    if mask:
      image = region_of_interest(image, vertices)  
    if image is not None:
      image = cv2.resize(image, (img_size,img_size))
      # img = cv2.resize(img, (300, 300))
      
      img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      graycom = feature.greycomatrix(img_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

      contrast = feature.greycoprops(graycom, 'contrast')
      dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
      homogeneity = feature.greycoprops(graycom, 'homogeneity')
      energy = feature.greycoprops(graycom, 'energy')
      correlation = feature.greycoprops(graycom, 'correlation')
      ASM = feature.greycoprops(graycom, 'ASM')



      gemeral_info_1 = [stat.skew(stat.skew(image[:, :, 0]))# find the RGB skew
                      , stat.skew(stat.skew(image[:, :, 1]))
                      , stat.skew(stat.skew(image[:, :, 2]))
                      , stat.kurtosis(stat.kurtosis(image[:, :, 0]))# find the RGB kurtosis
                      , stat.kurtosis(stat.kurtosis(image[:, :, 2]))
                      , stat.kurtosis(stat.kurtosis(image[:, :, 1]))
                      , stat.entropy(stat.entropy(image[:, :, 0]))# find the RGB entropy
                      , stat.entropy(stat.entropy(image[:, :, 1]))
                      , stat.entropy(stat.entropy(image[:, :, 2]))
                      , np.ptp(np.ptp((image[:, :, 0]),axis=1))# find the RGB ptp
                      , np.ptp(np.ptp((image[:, :, 1]),axis=1))
                      , np.ptp(np.ptp((image[:, :, 2]),axis=1))
                      , path[i]]
                            
      hist = ltp_hist(img_ltp, img_size, img_gray, hist_dim)
      gemeral_info_2 = general_img_info(image, contrast, dissimilarity, homogeneity, energy, correlation, ASM)
      h = np.append(hist, gemeral_info_2)
      g = np.append(h, gemeral_info_1)
      features.append(g)
    else:
      pass

  return features   




def potholes_predictions(images_path, model_path):
  features_list = features_extraction(images_path, 300, 119)
  loaded_model = pickle.load(open(model_path, 'rb'))
  X = np.array(features_list)[:, :-1].astype(float)
  X = np.nan_to_num(X, neginf=0.0,nan= 0.0, posinf = 0.0)#.astype(np.float64)

  pred = loaded_model.predict(X)
  df = pd.DataFrame()
  df['image'] = np.array(features_list)[:, -1]
  df['mask'] = np.array(features_list)[:, -1]
  df['prediction'] = pred
  for i in range(len(pred)):
    if df.prediction[i] == '1':
      print(f'A pothole(s) have been detected in the image with the path: {df.image[i]}')
    else:
      print('No pothole(s) have been detected ')
  return df








def conv2d_block(input_tensor, n_filters, kernel_size = 3, activation='relu', batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x



def upsampling(input_tensor, n_filters, concat_layer, activation='relu'):
    '''
    Block of Decoder
    '''
    x = Conv2DTranspose(n_filters, (3, 3), strides = (2, 2), padding = 'same')(input_tensor)
    x = concatenate([x, concat_layer])
    x = Dropout(0.2)(x)
    x = conv2d_block(x, n_filters, kernel_size = 3, activation=activation, batchnorm = True)
    return x


def DenseNet_UNET(n_filters, activation):


    ########################################################

                    #ENCODER#

    # Layer name of encoders to be concatenated
    names = ['pool3_pool', 'pool2_pool', 'pool1','conv1/relu']
    # Transfer learning approach without the classification head
    encoder = DenseNet201(include_top=False, weights='imagenet', input_shape=(320,320,3))
    for layer in encoder.layers:
      layer.trainable = True
      inputs = encoder.input
      x = encoder.output

                    #ENCODER#

    ########################################################


    ########################################################

                    #bottleneck#


    # decoder blocks linked with corresponding encoder blocks
    bneck = Conv2D(filters=n_filters*16, kernel_size=(1,1), padding='same')(x)
    x = Activation(activation)(bneck)                    

                    #bottleneck#

    ########################################################


    ########################################################

                    #DECODER#

    x = upsampling(input_tensor=bneck, n_filters=n_filters*16, activation=activation, concat_layer=encoder.get_layer(names[0]).output)
    x = upsampling(input_tensor=x, n_filters=n_filters*8, activation=activation, concat_layer=encoder.get_layer(names[1]).output)
    x = upsampling(input_tensor=x, n_filters=n_filters*4, activation=activation, concat_layer=encoder.get_layer(names[2]).output)
    x = upsampling(input_tensor=x, n_filters=n_filters*2, activation=activation, concat_layer=encoder.get_layer(names[3]).output)
    x = Conv2DTranspose(n_filters*2, (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Activation(activation)(x)
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

                    #DECODER#

    ########################################################

    model = Model(inputs=inputs, outputs=x)
    return model



class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, dataframe, col, batch_size, shuffle=False, dim=(320,320)):
    # for reproducibility
    np.random.seed(43)
    # dataframe containing the subset of image and depth pairs
    self.df = dataframe
    # chosen Height and Width of the RGB image
    self.height, self.width = dim
    # choice of shuffling the data
    self.shuffle = shuffle
    self.batch_size = batch_size
    # unique set of RGB images
    self.ids = dataframe['image'].unique()
    # Map the image with depth maps
    self.imgtodpth = dataframe.set_index('image')[col].to_dict()
    self.on_epoch_end()

  def __len__(self):
    '''
    Returns the length of dataset.
    '''
    return len(self.df) // self.batch_size

  def on_epoch_end(self):
    '''
    Shuffles the data at the end of every epoch
    '''
    self.indexes = np.arange(len(self.ids))
    if self.shuffle:
      np.random.shuffle(self.indexes)
  
  def __getitem__(self,index):
    '''
    returns the batch of image and depth pairs 
    '''
    # select the batch of pair indexes 
    idxs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
    # randomly select whether to flip the image
    flip = np.random.choice([True, False])
    # select the image id's for the above indexes
    query_imgs = [self.ids[idx] for idx in idxs]
    # select corresponding depth pair for the image
    target_imgs = [self.imgtodpth[img] for img in query_imgs]
    # preprocess the image 
    processed_query_img = self._preprocess_image(query_imgs, flip)
    # preprocess the depth map
    processed_depth_img = self._preprocess_depth(target_imgs, flip)
    return processed_query_img, processed_depth_img

  def _preprocess_image(self,images, flip):
    '''
    Resize, Normalize and randomly Augments the image set. 
    '''
    # placeholder for storing the processed images
    processed = []
    for img in images:
      # resize the image to 640x480
      resized_img = cv2.resize(cv2.imread(img),(self.height,self.width)).astype(np.float32)
      # normalize the image to {0,1}
      scaled_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
      # flip the image horizontally
      if flip:
        scaled_img = cv2.flip(scaled_img, 1)
      # finally append each image
      processed.append(scaled_img)
    return np.array(processed)

  def _preprocess_depth(self,images, flip):
    '''
    Resize, Normalize and randomly Augments the depth maps.
    '''
    # placeholder for storing the processed depth maps
    processed = []
    for img in images:
      # resize the depth map to 320x240
      resized_img = cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY),(320,320)).astype(np.float32)
      # normalize it to range {0,1}
      scaled_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
      # flip the image horizontally
      if flip:
        scaled_img = cv2.flip(scaled_img, 1)
      # add the color channel as cv2 grayscale image doesnt contain color channel but tensorflow expects it
      scaled_img = np.expand_dims(scaled_img, axis=-1)
      # finally append the image
      processed.append(scaled_img)
    return np.array(processed)
    






def loss_function(y_true, y_pred):

  #Cosine distance loss
  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
  
  # edge loss for sharp edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
  
  # structural similarity loss
  l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

  # weightage
  w1, w2, w3 = 1.0, 1.0, 0.1
  return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


def accuracy_function(y_true, y_pred):
  return K.mean(K.equal(K.round(y_true), K.round(y_pred)))



def get_df(path):

    image = [f'{path}/rgb/{i}' for i in next(os.walk(f'{path}/rgb'))[2]]
    tdisp_map = [f'{path}/tdisp/{i}' for i in next(os.walk(f'{path}/tdisp'))[2]]
    mask = [f'{path}/label/{i}' for i in next(os.walk(f'{path}/label'))[2]]

    df = pd.DataFrame(columns=['image', 'depth', 'mask'])
    df['image'] = image
    df['depth'] = tdisp_map
    df['mask'] = mask
    return df


def plot_3d_prediction(depth, mask):
  # img_org = cv2.imread(depth_pred[20]*255)
  img_org = depth
  img_org = cv2.cvtColor(img_org, cv2.COLOR_GRAY2BGR)

  img_mask_gray = mask
  # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
  img_mask = cv2.cvtColor(img_mask_gray, cv2.COLOR_GRAY2BGR)
  ##Resizing images
  img_org = cv2.resize(img_org, (400,400), interpolation = cv2.INTER_AREA)
  img_mask = cv2.resize(img_mask, (400,400), interpolation = cv2.INTER_AREA)

  # array = np.array(img_mask_gray*255, np.uint8)

  # (cnt, hierarchy) = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  # print(f'There are {len(cnt)} pothole(s) in the image')
  img_org = cv2.bitwise_and(img_mask, img_org)
  img_org = img_org*255


  y = range( img_org.shape[0] )
  x = range( img_org.shape[1] ) 
  X, Y = np.meshgrid(x, y)

  Z = img_org[:,:,0]
  print('Predicted depth map + predicted mask: \n\n')
  print('Estimated Width In px:', sum((mask*255 > 1).any(axis=0))[0])
  print('Estimated Height In px:', sum((mask *255> 1).any(axis=1))[0])
  print('Estimated Depth In px: ', round(Z.max() - Z.min(), 2))
  print('Estimated Volume In px:', round(sum((mask*255 > 1).any(axis=0))[0] * sum((mask *255> 1).any(axis=1))[0] *  Z.max() - Z.min(), 2))

  new_x = np.zeros((400, 400))
  new_z = np.zeros((400, 400))
  new_y = np.zeros((400, 400))


  for i in range(400):
    for j in range(400):
      if Z[i, j] < 1 :
        new_x[i, j]=np.NaN
        new_z[i, j] = np.NaN
        new_y[i, j] = np.NaN
      else:
        new_x[i, j]=X[i, j]
        new_z[i, j] = Z[i, j]
        new_y[i, j] = Y[i, j]

  fig = go.Figure(data=[go.Surface(z=-new_z, x=-new_x, y=-new_y)])
  fig.update_layout(title='3-D Scene Of The Pothole', autosize=False,
                    width=500, height=500,
                    margin=dict(l=65, r=50, b=65, t=90))
  fig.show()


