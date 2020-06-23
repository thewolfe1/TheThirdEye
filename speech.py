import numpy as np
import glob

from keras.backend import flatten

from preprocess import audio2vector, audio2vector2, rotateAudio, addNoiseAudio, resizeAudio, removePartAudio, slowAudio, \
    fastAudio

from keras import metrics
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization,Conv1D,GlobalAveragePooling2D


class Speech:
    def __init__(self):
        """
        Constructor

        """
        self.data = []
        self.labels = []
        self.labels_dict = {'Noise':0}
        self.proccesed_data = []

    def expend_data(self):
        """
        The function expends the audio files.

        """
        rotateAudio('speech')
        addNoiseAudio('speech')
        resizeAudio('speech')
        removePartAudio('speech')
        slowAudio('speech')
        fastAudio('speech')

    def read_data(self):
        """
        read labels file and save as a list
        """
        print('reading...')
        with open('Files/Names.txt') as fp:
            for line in enumerate(fp):
                self.labels.append(line[1].rstrip())
                self.data.append(glob.glob('speech/{}/*.wav'.format(line[1].strip())))
                self.data.append(glob.glob('speech/{}/noise/*.wav'.format(line[1].strip())))
                self.data.append(glob.glob('speech/{}/remove/*.wav'.format(line[1].strip())))
                self.data.append(glob.glob('speech/{}/resize/*.wav'.format(line[1].strip())))
                self.data.append(glob.glob('speech/{}/rotate/*.wav'.format(line[1].strip())))
                self.data.append(glob.glob('speech/{}/faster/*.wav'.format(line[1].strip())))
                self.data.append(glob.glob('speech/{}/slower/*.wav'.format(line[1].strip())))
        self.data.append(glob.glob('recordings/Noise/*.wav'))
        # flatten data
        self.data = [item for sublist in self.data for item in sublist]
        print('done')


    def preprocess_labels(self):
        """
        preprocess the labels with to_categorical
        """
        count = 1
        temp = []

        for i in self.labels:
            self.labels_dict[i] = count
            count += 1
        for i in range(len(self.data)):
            for j in self.labels:
                if (j in self.data[i]):
                    temp.append(self.labels_dict[j])
                    break
                if 'Noise' in self.data[i]:
                    print(self.data[i])
                    temp.append(self.labels_dict['Noise'])
                    break
        with open('speech.csv', 'w') as f:
            for key in self.labels_dict.keys():
                f.write("%s,%s\n" % (key, self.labels_dict[key]))
        self.encoded = to_categorical(temp)


    def preprocess_data(self):
        """
        preprocess the data by extracting features
        """
        for i in self.data:
            self.proccesed_data.append(audio2vector(i))
        self.proccesed_data = np.asarray(self.proccesed_data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.proccesed_data, self.encoded, test_size=0.05, random_state=42)


    def train(self,rows,columns,channels,epochs):
        """
        train the model
        """

        self.X_train = self.X_train.reshape(int(self.X_train.shape[0]), rows, columns, channels)
        self.X_test = self.X_test.reshape(int(self.X_test.shape[0]), rows, columns, channels)


        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=2, input_shape=(rows, columns, channels), padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=1))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(self.encoded.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'mse'])
        model.fit(self.X_train, self.y_train, validation_split=0.05, epochs=epochs, batch_size=32, validation_data=(self.X_test, self.y_test))

        model.save('model.h5')
        model.save_weights('model_weight.h5')

        loss, acc, mse = model.evaluate(self.X_test, self.y_test, verbose=0)

        print(mse)
        score = model.evaluate(self.X_train, self.y_train, verbose=0)
        print("Training Accuracy: ", score[1] * 100)
        print("Training mse: ", score[2] )

        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Testing Accuracy: ", score[1] * 100)
        predictions = model.predict(self.X_test, batch_size=1)

        print("Testing mse: ", score[2] )

        print(classification_report(self.y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=list(self.labels_dict.keys())))


