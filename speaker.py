import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import librosa
from librosa.display import cmap
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
import keras
import warnings
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
import split_folders
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



class Speaker:
    def __init__(self,path='speaker/'):
        """
        Constructor

        Parameters:
            path (string): audio files path.
        """
        self.path = path
        self.labels = os.listdir(self.path)

    def audio_to_image(self):
        """
        convert all the files to images
        """
        for l in self.labels:
            pathlib.Path(f'img_data/{l}').mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(f'./{self.path}{l}'):
                songname = f'./{self.path}{l}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=5)
                print(y.shape)
                plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB')
                plt.axis('off')
                plt.savefig(f'img_data/{l}/{filename[:-3].replace(".", "")}.png')
                plt.clf()


    def preprocess(self):
        """
        reshape the images and split
        """
        split_folders.ratio('./img_data/', output="./data", seed=1337, ratio=(.7, .3))  # default values

        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.training_set = train_datagen.flow_from_directory('./data/train', target_size=(64, 64), batch_size=32,
                                                         class_mode='categorical', shuffle=False)
        self.test_set = test_datagen.flow_from_directory('./data/val', target_size=(64, 64), batch_size=32,
                                                    class_mode='categorical', shuffle=False)


    def train(self,steps=100,val_steps=200,epochs=50,learning_rate=0.01,momentum=0.9):
        """
        train the model
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=(64, 64, 3)))
        model.add(AveragePooling2D((2, 2), strides=(2, 2)))
        model.add(Activation('relu'))  # 2nd hidden layer
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(AveragePooling2D((2, 2), strides=(2, 2)))
        model.add(Activation('relu'))  # 3rd hidden layer
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(AveragePooling2D((2, 2), strides=(2, 2)))
        model.add(Activation('relu'))  # Flatten
        model.add(Flatten())
        model.add(Dropout(rate=0.5))  # Add fully connected layer.
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.5))  # Output layer
        model.add(Dense(len(self.labels)))
        model.add(Activation('softmax'))
        model.summary()

        sgd = SGD(lr=learning_rate, momentum=momentum, decay=learning_rate / epochs, nesterov=False)
        model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy', 'mse'])

        model.fit_generator(self.training_set, steps_per_epoch=steps, epochs=epochs, validation_data=self.test_set,validation_steps=val_steps)

        model.save('model2.h5')
        model.save_weights('model_weight2.h5')

        score = model.evaluate(self.training_set, verbose=0)
        print("Training Accuracy: ", score[1] * 100)
        print("Training mse: ", score[2])
        score = model.evaluate(self.test_set, verbose=0)
        print("Testing Accuracy: ", score[1] * 100)
        print("Testing mse: ", score[2])
