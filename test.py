from __future__ import print_function
#cambio
"Fingerprint Image denoising using M-net"

import os
import glob
import numpy as np
import PIL
from PIL import Image
import datetime
import scipy as sp
import imageio
import sys
import keras as keras
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.merge import concatenate
#from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
from keras import backend as K
from tensorflow.keras.optimizers import SGD

import math

# image_data_format = channels_last
K.set_image_data_format('channels_last')

#Necesario para que no piense que estamos haciendo un Decompression bomb DOS attack:
#Establecemos el tamaño maximo de pixeles en el mayor de todas nuestras imagenes:
PIL.Image.MAX_IMAGE_PIXELS = 248500000

# Parameter
##############################################################
modelName = 'FPD_M_net_weights'
lrate = 0.1
decay_Rate = 1e-6
height = 400
width = 275
padSz = 88
d = 8
padX = (d - height%d)
padY = (d - width%d)
ipDepth = 3
ipHeight = height + padX + padSz
ipWidth = width + padY + padSz
outDepth = 1
is_zeroMean = False
##############################################################

# Loss function
def my_loss(y_true, y_pred):
    l1_loss = K.mean(K.abs(y_pred - y_true))
    return l1_loss


# Define the neural network
def getFPDMNet(patchHeight, patchWidth, ipCh, outCh):

    # Input
    input1 = Input((patchHeight, patchWidth, ipCh))

    # Encoder
    conv1 = Conv2D(16, (3, 3), padding='same')(input1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv1 = concatenate([input1, conv1], axis=-1)
    conv1 = Conv2D(16, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    #
    input2 = MaxPooling2D(pool_size=(2, 2))(input1)
    conv21 = concatenate([input2, pool1], axis=-1)

    conv2 = Conv2D(32, (3, 3), padding='same')(conv21)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)

    conv2 = concatenate([conv21, conv2], axis=-1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #
    input3 = MaxPooling2D(pool_size=(2, 2))(input2)
    conv31 = concatenate([input3, pool2], axis=-1)

    conv3 = Conv2D(64, (3, 3), padding='same')(conv31)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)
    
    conv3 = concatenate([conv31, conv3], axis=-1)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    #
    input4 = MaxPooling2D(pool_size=(2, 2))(input3)
    conv41 = concatenate([input4, pool3], axis=-1)

    conv4 = Conv2D(128, (3, 3), padding='same')(conv41)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)
    
    conv4 = concatenate([conv41, conv4], axis=-1)
    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)

    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder
    conv5 = UpSampling2D(size=(2, 2))(conv4)
    conv51 = concatenate([conv3, conv5], axis=-1)

    conv5 = Conv2D(64, (3, 3), padding='same')(conv51)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.2)(conv5)
    
    conv5 = concatenate([conv51, conv5], axis=-1)
    conv5 = Conv2D(64, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    #
    conv6 = UpSampling2D(size=(2, 2))(conv5)
    conv61 = concatenate([conv2, conv6], axis=-1)

    conv6 = Conv2D(32, (3, 3), padding='same')(conv61)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.2)(conv6)
    
    conv6 = concatenate([conv61, conv6], axis=-1)
    conv6 = Conv2D(32, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    #
    conv7 = UpSampling2D(size=(2, 2))(conv6)
    conv71 = concatenate([conv1, conv7], axis=-1)

    conv7 = Conv2D(16, (3, 3), padding='same')(conv71)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)
    
    conv7 = concatenate([conv71, conv7], axis=-1)
    conv7 = Conv2D(16, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    # Final
    conv81 = UpSampling2D(size=(8, 8))(conv4)
    conv82 = UpSampling2D(size=(4, 4))(conv5)
    conv83 = UpSampling2D(size=(2, 2))(conv6)
    conv8 = concatenate([conv81, conv82, conv83, conv7], axis=-1)
    conv8 = Conv2D(outCh, (1, 1), activation='sigmoid')(conv8)

    ############
    model = Model(inputs=input1, outputs=conv8)

    sgd = SGD(lr=lrate, decay=decay_Rate, momentum=0.75, nesterov=True)
    model.compile(optimizer=sgd, loss=my_loss)

    return model

####################### Load & Preprocess Data #######################

# pad Data
def padData(img):

    # For grayscale image
    if len(img.shape) == 3:
        H, W, nB = img.shape
        img = np.transpose(img, (2,1,0))
        img = np.reshape(img, (nB, W, H, 1))
    
    # Pad X, Y such that it will be divisible by 8
    nB, H, W, D = img.shape
    temp = np.zeros((nB, H + padX, W + padY, D))
    temp[:, :H, :W, :] = img
    nB, H, W, D = temp.shape

    # Pad extra for network
    img = np.zeros((nB, H + padSz, W + padSz, D))
    for i in range(0, nB):
        for j in range(0, D):
            img[i,:,:,j] = np.lib.pad(temp[i,:,:,j], (int(padSz/2)), 'edge')

    img = (img.astype('float32'))
    return img

def load_data(dataPath):
    imgsX = imread(dataPath)

    if is_zeroMean:
        imgsX = np.array(imgsX)/127.5 - 1.
    else:
        imgsX = np.array(imgsX)/255.0

    # Pad images
    imgsX = padData(imgsX)
    return imgsX

def imread(path, is_gt=False):
    temp = imageio.imread(path).astype(float)
    if is_gt:
        temp = np.expand_dims(temp, -1)
    temp = np.expand_dims(temp, 0)
    return temp

#CREAMOS UNA FUNCION QUE CONVIERTA LOS ARCHIVOS A FORMATO JPG (es el que acepta la red):

def ConvertToJPG(dataPath):
    origen_arch = os.path.join(dataPath, "ImagenesTFG")

    files = os.listdir(origen_arch)
    for fname in files:
        #print('convirtiendo:' + fname)
        file_name     = fname.split('.')[0]
        new_file_name = file_name + '.jpg'
        new_destino   = os.path.join(dataPath, new_file_name)

        ubi_img = os.path.join(origen_arch, fname)
        img = Image.open(ubi_img)
        rgb_im = img.convert('RGB')
        rgb_im.save(new_destino)

#Funcion que prepara la imagen grande para que sea divisible en parches de tamaño 496x368
def prepararParches(img):
    if len(img.shape) == 3:
        H, W, nB = img.shape
        img = np.transpose(img, (2,1,0))
        img = np.reshape(img, (nB, W, H, 1))

    d_h = 496
    height = img.shape[1]
    padX = (d_h - height%d_h)
    print(padX)

    d_w = 368
    width = img.shape[2]
    padY = (d_w - width%d_w)
    print(padY)

    nB, H, W, D = img.shape
    temp = np.zeros((nB, H + padX, W + padY, D))
    temp[:, :height, :width, :] = img
    print(temp.shape)
    return temp

####################################

def test_FPDMNet(dataPath):
    #PRIMERO: Convertimos todas las imagenes añadidas por nosotros a formato .jpg:
    ConvertToJPG(dataPath)

    # Get FPD_M_Net
    loadWeightsPath = os.path.join("./weights", modelName + ".hdf5")
    fpDenNet = getFPDMNet(ipHeight, ipWidth, ipDepth, outDepth)

    # load weights
    fpDenNet.load_weights(loadWeightsPath)

    imgDataFiles     = glob.glob(os.path.join(dataPath, '*.jpg'))
    imgDataFiles.sort()

    savePath = os.path.join(dataPath, "Results")
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for i in range(0, len(imgDataFiles)):

        print('-----------------------------> Test: ', i, ' <-----------------------------')

        # Read fundus data from imgDataFiles
        imTest = load_data(imgDataFiles[i])

        if imTest.shape[1] > 496 or imTest.shape[2] > 368:
            imTest = prepararParches(imTest)
            print(imTest.shape)
            #Partimos en parches, los pasamos uno a uno por la red:
            #Se llama a .predict desde aqui:
            it_i = 0
            it_j = 0
            #Creamos una carpeta por cada imagen grande donde se guardaran sus parches:
            file_name     = os.path.basename(imgDataFiles[i])
            file_name     = file_name.split('.')[0]
            savePath_temp = os.path.join(savePath, file_name)
            if not os.path.exists(savePath_temp):
                os.makedirs(savePath_temp)

            head_i = '#'
            head_j = '_'

            while it_i*496 < imTest.shape[1]:
                #print("it_i = ", it_i)
                it_j  = 0

                decenas_i = it_i/10
                decenas_i_f = int(math.floor(decenas_i))
                for aux in range(0,decenas_i_f):
                  head_i = head_i + '#'

                while it_j*368 < imTest.shape[2]:
                    parche = imTest[:, (it_i*496):((it_i+1)*496), (it_j*368):((it_j+1)*368),:]
                    #print("it_j = ", it_j)
                    
                    decenas_j = it_j/10
                    decenas_j_f = int(math.floor(decenas_j))
                    for aux in range(0,decenas_j_f):
                      head_j = head_j + '_'


                    print(parche.shape)
                    imPred = fpDenNet.predict(parche, batch_size=1)
                    imPred = np.squeeze(imPred)
                    unpad = int(padSz/2)
                    imPred = imPred[unpad:-unpad, unpad:-unpad]
                    imPred = imPred[:-padX, :-padY]
                    print('After imPred: ', imPred.shape)
                    file_name     = os.path.basename(imgDataFiles[i])
                    file_name     = file_name.split('.')[0]+ head_i + str(it_i) + head_j + str(it_j) + '.jpg'
                    print("Transformando: ",file_name)

                    Image.fromarray(np.uint8(imPred * 255)).save(os.path.join(savePath_temp, os.path.basename(file_name)))
                    it_j += 1
                    head_j = '_'
                it_i += 1
                head_i = '#'

            #Vamos a unir todos los parches (su resultado despues de pasar por la red)
            div_i = int(imTest.shape[1]/496)
            div_j = int(imTest.shape[2]/368)
            print("div_i = ",div_i)
            print("div_j = ",div_j)
            it_i = 0
            it_j = 0
            itt  = 0
            H = 400
            W = 275
            salida = np.zeros((H*div_i, W*div_j))
            #print(salida.shape)
            files = os.listdir(savePath_temp)
            files.sort()
            
            for it_i in range(0,div_i):
                #print(fname)
                #it_j  = 0
                print("it_i =", it_i)
                for it_j in range (0,div_j):
                    fname = files[itt]
                    sp_temp = os.path.join(savePath_temp, fname)
                    img = imageio.imread(sp_temp)
                    salida[(it_i*400):((it_i+1)*400), (it_j*275):((it_j+1)*275)] = img
                    #print("it_j= ", it_j)
                    #i += 1
                    #it_j += 1
                    itt += 1
            #it_i += 1
            Image.fromarray(np.uint8(salida)).save(os.path.join(savePath, os.path.basename(imgDataFiles[i])))
        else:
            imPred = fpDenNet.predict(imTest, batch_size=1)
            #print('Before imPred: ', imPred.shape)

            imPred = np.squeeze(imPred)
            unpad = int(padSz/2)
            imPred = imPred[unpad:-unpad, unpad:-unpad]
            imPred = imPred[:-padX, :-padY]
            #print('After imPred: ', imPred.shape)
            Image.fromarray(np.uint8(imPred * 255)).save(os.path.join(savePath, os.path.basename(imgDataFiles[i])))

if __name__ == '__main__':
    dataPath = sys.argv[1]
    #dataPath = './test'

    test_FPDMNet(dataPath)



