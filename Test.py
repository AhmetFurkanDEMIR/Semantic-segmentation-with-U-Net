import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
import cv2

class automaticmaplabelling():

    def __init__(self,modelPath,full_chq,imagePath,width,height,channels):

        print (modelPath)
        print(imagePath)
        print(width)
        print(height)
        print(channels)
        self.modelPath=modelPath
        self.full_chq=full_chq
        self.imagePath=imagePath
        self.IMG_WIDTH=width
        self.IMG_HEIGHT=height
        self.IMG_CHANNELS=channels
        self.model = self.U_net()
        
    def mean_iou(self,y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def U_net(self):

        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

        c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
        c6 = Dropout(0.3) (c6)
        c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

        c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
        c7 = Dropout(0.3) (c7)
        c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
        u6 = concatenate([u6, c4])
        c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
        c8 = Dropout(0.2) (c8)
        c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
        u7 = concatenate([u7, c3])
        c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
        c9 = Dropout(0.2) (c9)
        c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c9)
        u8 = concatenate([u8, c2])
        c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
        c10 = Dropout(0.1) (c10)
        c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c10)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c10)
        u9 = concatenate([u9, c1], axis=3)
        c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
        c11 = Dropout(0.1) (c11)
        c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c11)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[self.mean_iou])
        model.load_weights(self.modelPath)
        model.summary()
        return model

    def prediction(self):

        img=cv2.imread(self.imagePath,0)
        img=np.expand_dims(img,axis=-1)
        x_test= np.zeros((1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        img=resize(img,(self.IMG_HEIGHT,self.IMG_WIDTH),mode='constant',preserve_range=True)
        x_test[0]=img
        preds_test= self.model.predict(x_test, verbose=1)
        
        preds_test = (preds_test > 0.7).astype(np.uint8)
        mask=preds_test[0]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 1:
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0

        return x_test[0],mask

def main():
    test_image_name = "test.jpeg"
    automaticmaplabellingobj= automaticmaplabelling('model-dsbowl2018-1.h5',True,test_image_name,256,256,3)
    testimg,mask = automaticmaplabellingobj.prediction()
    print('Showing images..')
    cv2.imshow('img',testimg)
    dim = (256, 256)
    resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imshow('mask',mask)

if __name__ == "__main__":
    main()