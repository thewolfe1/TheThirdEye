import csv
import multiprocessing
import pickle
import threading
import time
import os.path

import PyQt5
import tensorflow as tf

from PyQt5.QtCore import pyqtSlot, QThread
from PyQt5.QtWidgets import QMessageBox
from tensorflow.keras.models import load_model
from Utils import record
import queue
import logging


from preprocess import audio2vector,audio2Image
from Utils import lower_sound
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

signal = False
do_run = True
Queue = queue.Queue(20)
flag=1
CATEGORIES =  ['Tal', 'Olga','Semion','Alex','jackson','yweweler','theo','nicolas','FTBR0','FVFB0','FVMH0','MCPM0','MDAC0','MDPK0','MEDR0','MGRL0','MJEB1','MJWT0','MKLS0','MKLW0','MMGG0',
                'MMRP0','MPGH0','MPGR0','MPSW0','MRAI0','MRCG0','MRDD0','MRSO0','MRWS0','MTJS0','MTPF0','MTRR0','MWAD0','MWAR0','FCJF0','FDAW0','FDML0','FECD0','FETB0','FJSP0','FKFB0',
                'FMEM0','FSAH0','FSJK1','FSMA0']


model = load_model('model.h5',compile=False)
model.load_weights('model_weight.h5')
model2 = load_model('model2.h5',compile=False)
model2.load_weights('model_weight2.h5')
model._make_predict_function()
model2._make_predict_function()

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)



class ProducerThreadRecored(QThread):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, interval=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        super().__init__()
        self.interval = interval
        self.status = True
        self.runs = True
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True  # Daemonize thread

        #thread.start()  # Start the execution

    def setStatus(self, sts):
        self.status = sts

    def stop(self):
        self.status = False

    @pyqtSlot()
    def run(self):
        count = 0
        global Queue
        """ Method that runs forever """
        while self.status:
            record(count)
            Queue.put(count)
            logging.info("Produced " + str(count))
            count = count + 1
            if count == 20:
                count = 0
            time.sleep(self.interval)
        if signal:
            self.stop()



def pro_line(line):
    return "FOO: %s" % line





class ConsumerThreadRecordTest(QThread):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval
        thread = threading.Thread(target=self.run, args=())
        self.name = ''
        #self.signal = False
        thread.daemon = True  # Daemonize thread
        #thread.start()  # Start the execution

    temp = False


    def updateName(self, string):
            self.name = string


    def matchName(self):
        if self.foundMatch():
            name = self.name
            return name


    def foundMatch(self):
        self.temp = True;

    def updateStatus(self, boolean):
        self.foundMatch()

   # def setSignal(self,bool):
    #    self.signal = bool

    #def getSignal(self):
    #    return self.signal

    @pyqtSlot()
    def run(self):
        global Queue
        global flag
        global volume
        global signal
        while True:
            num = Queue.get()
            file_name = 'temp/output' + str(num) + '.wav'
            audio2Image('Tal', file_name, 0)
            train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            training_set = train_datagen.flow_from_directory('./data/test', target_size=(64, 64), batch_size=32,
                                                             class_mode='categorical', shuffle=False)

            pred = model2.predict_generator(training_set, steps=10, verbose=1)
            filenames = training_set.filenames
            predicted_class_indices = np.argmax(pred, axis=1)
            labels = (training_set.class_indices)
            labels = dict((v, k) for k, v in labels.items())
            predictions = [labels[k] for k in predicted_class_indices]
            predictions = predictions[:len(filenames)]
            print(filenames)
            print(predictions)
            while not os.path.exists(file_name):
                True
            with open(file_name) as source_file:
                # chunk the work into batches of 4 lines at a time

                #speaker model
                num_rows = 10
                num_columns = 4
                num_channels = 1
                file = audio2vector(file_name)
                file = file.reshape(1, num_rows, num_columns, num_channels)

                reader = csv.reader(open('dictonary.csv', 'r'))
                labels_dict = {}
                for row in reader:
                    k, v = row
                    labels_dict[k] = v

                reader = csv.reader(open('speech.csv', 'r'))
                labels_dict2 = {}
                for row in reader:
                    k, v = row
                    labels_dict2[k] = v
                """

                #word model
                true_clip = "recordings/exp/0.wav"
                true_mfcc_batch = get_clip_batch(true_clip, display=True)
                test_clip_1a = file_name
                test_mfcc_batch_1a = get_clip_batch(test_clip_1a, display=True)
                y_pred=model.predict([true_mfcc_batch, test_mfcc_batch_1a])
                """
                pred=''
                y_pred=1
                prediction = model.predict(file)
                item_prediction = model.predict_classes(file)
                for i, j in labels_dict2.items():
                    if j == str(item_prediction[0]):
                        pred=i
                        print('is keyword? {}'.format(i))
                #item_prediction = model2.predict_classes(file)
                if pred == 'yes':
                    prediction = model2.predict(file)
                    item_prediction = model2.predict_classes(file)
                    for i, j in labels_dict.items():
                        if j == str(item_prediction[0]):
                            print('speaker: {}'.format(i))
                            self.name = i
                            signal = True
                            break
                            all_processes = []



                var=''
                if var=='fds':
                    name = ''
                    try:
                        prediction = model2.predict(file)
                        item_prediction = model2.predict_classes(file)
                        for i, j in labels_dict.items():


                            if j == str(item_prediction[0]):

                                print('prediction: {}'.format(i))
                                name = i               # here i tried to get the name
                        #lower_sound(True)
                    except os.error as e:
                        None

                    self.updateStatus(True)
                    self.updateName(name)


                #print('processed')
                #pro_line(source_file)

            Queue.task_done()
            i = str(num)
            #os.remove(file_name)
            logging.info("Consumed " + str(num))
            time.sleep(self.interval)






