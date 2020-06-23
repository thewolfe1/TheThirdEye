import csv

import Utils
import queue
import time
import pymsgbox
import sys
import pyaudio
import wave
import pathlib
import os
import logging
import threading
import threadRecord
from speaker import Speaker
from preprocess import audio2Image, audio2vector
from speech import Speech
import qdarkstyle
from PyQt5.QtWidgets import QMessageBox, QPushButton, QLineEdit, QLabel, QGridLayout
from PyQt5 import QtCore, QtGui, QtWidgets
from database import Db
from Utils import record
from tensorflow.keras.models import load_model


from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class MainWindow(QtWidgets.QWidget):
    """
    Main Window page
    """
    switch_window = QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('Main Window')
        self.setObjectName("TheThirdEye")
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QtGui.QIcon('eye-512.png'))
        self.setFixedSize(490, 440)


        # -------------------------------------
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 531, 420))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()  # tab1 Code
        self.tab.setObjectName("tab")
        self.pushButtonRecord = QtWidgets.QPushButton(self.tab)
        self.pushButtonRecord.setGeometry(QtCore.QRect(10, 70, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonRecord.setFont(font)
        self.pushButtonRecord.setCheckable(False)
        self.pushButtonRecord.setObjectName("pushButtonRecord")
        self.checkBoxRecord = QtWidgets.QCheckBox(self.tab)
        self.checkBoxRecord.setGeometry(QtCore.QRect(10, 100, 111, 61))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.checkBoxRecord.setFont(font)
        self.checkBoxRecord.setCheckable(True)
        self.checkBoxRecord.setTristate(False)
        self.checkBoxRecord.setObjectName("checkBoxRecord")


        self.labelNameDisplay = QtWidgets.QLabel(self.tab)
        self.labelNameDisplay.setGeometry(QtCore.QRect(10, 200, 101, 16))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(9)
        self.labelNameDisplay.setFont(font)
        self.labelNameDisplay.setObjectName("labelNameDisplay")

        self.NameDisplayText = QtWidgets.QTextBrowser(self.tab)
        self.NameDisplayText.setGeometry(QtCore.QRect(111, 195, 121, 30))
        self.NameDisplayText.setObjectName("NameDisplayText")

        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()  # Tab2 Code
        self.tab_2.setObjectName("tab_2")

        self.labelChangeName = QtWidgets.QLabel(self.tab_2)
        self.labelChangeName.setGeometry(QtCore.QRect(40, 60, 191, 41))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.labelChangeName.setFont(font)
        self.labelChangeName.setObjectName("labelChangeName")

        self.labelAddPeople = QtWidgets.QLabel(self.tab_2)
        self.labelAddPeople.setGeometry(QtCore.QRect(40, 145, 170, 41))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.labelAddPeople.setFont(font)
        self.labelAddPeople.setObjectName("labelAddPeople")

        self.labelAddRecording = QtWidgets.QLabel(self.tab_2)
        self.labelAddRecording.setGeometry(QtCore.QRect(40, 180, 151, 31))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.labelAddRecording.setFont(font)
        self.labelAddRecording.setObjectName("labelAddRecording")

        self.labelAddNewRecording = QtWidgets.QLabel(self.tab_2)
        self.labelAddNewRecording.setGeometry(QtCore.QRect(40, 240, 190, 31))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.labelAddNewRecording.setFont(font)
        self.labelAddNewRecording.setObjectName("labelAddNewRecording")

        self.lineEditNameChange = QtWidgets.QLineEdit(self.tab_2)
        self.lineEditNameChange.setGeometry(QtCore.QRect(240, 70, 121, 31))
        self.lineEditNameChange.setObjectName("lineEditNameChange")

        self.lineEditAddPeople = QtWidgets.QLineEdit(self.tab_2)
        self.lineEditAddPeople.setGeometry(QtCore.QRect(170, 155, 120, 30))
        self.lineEditAddPeople.setObjectName("lineEditAddPeople")

        self.pushButtonAddPeople = QtWidgets.QPushButton(self.tab_2)
        self.pushButtonAddPeople.setGeometry(QtCore.QRect(308, 155, 93, 28))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.pushButtonAddPeople.setFont(font)
        self.pushButtonAddPeople.setObjectName("pushButtonAddPeople")

        self.pushButtonChangeName = QtWidgets.QPushButton(self.tab_2)
        self.pushButtonChangeName.setGeometry(QtCore.QRect(380, 70, 93, 28))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        self.pushButtonChangeName.setFont(font)
        self.pushButtonChangeName.setObjectName("pushButtonChangeName")



        self.commandLinkButtonNewRecording = QtWidgets.QCommandLinkButton(self.tab_2)
        self.commandLinkButtonNewRecording.setGeometry(QtCore.QRect(380, 240, 90, 40))
        self.commandLinkButtonNewRecording.setObjectName("commandLinkButtonNewRecording")

        self.lineEditAddNameTag = QtWidgets.QLineEdit(self.tab_2)
        self.lineEditAddNameTag.setGeometry(QtCore.QRect(240, 245, 120, 30))
        self.lineEditAddNameTag.setObjectName("lineEditAddNameTag")

        self.commandLinkButtonLogout = QtWidgets.QCommandLinkButton(self.tab_2)
        self.commandLinkButtonLogout.setGeometry(QtCore.QRect(380, 300, 90, 40))
        self.commandLinkButtonLogout.setObjectName("commandLinkButtonLogout")

        self.commandLinkButtonTrainModel = QtWidgets.QCommandLinkButton(self.tab_2)
        self.commandLinkButtonTrainModel.setGeometry(QtCore.QRect(360, 350, 120, 40))
        self.commandLinkButtonTrainModel.setObjectName("commandLinkButtonTrainModel")

        self.tabWidget.addTab(self.tab_2, "")



        self.retranslateUi(self)
        self.tabWidget.setCurrentIndex(0)

        # Connecting Buttons to functions:

        self.pushButtonRecord.clicked.connect(self.checkBoxRecord.animateClick)
        self.pushButtonRecord.clicked.connect(self.startRecording)


        text = open("Files/ID_Name.txt").read()
        self.NameDisplayText.setPlainText(text)

        self.pushButtonChangeName.clicked.connect(self.editUserName)

        self.pushButtonAddPeople.clicked.connect(self.addPeople)


        self.commandLinkButtonNewRecording.clicked.connect(self.createRecord)

        self.commandLinkButtonLogout.clicked.connect(self.logoutGUI)

        self.commandLinkButtonTrainModel.clicked.connect(self.TrainModel)

        QtCore.QMetaObject.connectSlotsByName(self)

    def logoutGUI(self):   #goes back to the login page
        self.switch_window.emit()

    def TrainModel(self):
        '''
        Trains the model wit given recordings
        :param recordings from
        '''
        
        model = Speech()
        model.expend_data()
        model.read_data()
        model.preprocess_labels()
        model.preprocess_data()
        model.train(10, 4, 1, 1000)

        model2 = Speaker()
        model2.audio_to_image()
        model2.preprocess()
        model2.train()

    def startRecording(self):

        '''
        Starts recording and activated the two threads
        '''

        if threadRecord.signal:
            self.pushButtonRecord.setEnabled(False)
            # self.thread1.setStatus(False)
            # QMessageBox.about(self, "Title", self.thread2.name)

        elif not self.checkBoxRecord.isChecked():
            self.thread1 = Thread1(self)
            self.thread2 = Thread2(self)
            # if self.thread2.getSignal()==False:
            self.thread1.start()
            self.thread2.start()


        else:
            self.pushButtonRecord.setEnabled(False)
            # self.thread1.setStatus(False)

            folder = 'temp'
            # while not self.thread1.running and not self.thread2.running:

            #    QMessageBox.about(self, "Title", "Thread 1 stopped")  # popup of recognition
            self.thread1.stop()
            self.thread1.sleep(1)
            if self.thread2.running:
                self.thread2.stop()

            if not self.thread1.running and not self.thread2.running:
                print("Nothing Runs")
                # QMessageBox.about(self, "Title", "Threads stopped")  # popup of recognition
                '''
            while not self.thread1.running:
                try:
                    lower_sound(False)
                    flag = 0
                    shutil.rmtree(folder)
                except WindowsError:
                    flag = 1
                '''
            self.pushButtonRecord.setEnabled(True)


    def editUserName(self):
        """
        Edits the username
        :param with given string changes the username
        """
        value = self.lineEditNameChange.text()
        f = open("Files/ID_Name.txt", "w")
        f.write(str(value))
        f.close()

        text = open('Files/ID_Name.txt').read()
        self.NameDisplayText.setPlainText(text)

    def addPeople(self):
        """
        Adds New People name for detection

        """
        value = self.lineEditAddPeople.text()
        f = open("Files/People_Names.txt", "a")
        f.write(str(value + "\n"))
        f.close()



    def createRecord(self):
        value = self.lineEditAddNameTag.text()
        self.createNewRecording(value)

    def createNewRecording(self, string):
        """
        Creates new recording and saves it
        :param string: identifies whos recording is this
        """
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 2
        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        print("recording...")
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
        msg.setWindowTitle("Recording")
        msg.setText("After pressing OK the program will start recording...")
        x = msg.exec_()  # this will show our message box

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")

        # stop Recording
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
        msg.setWindowTitle("Finished")
        msg.setText("Finished Recording")
        x = msg.exec_()  # this will show our message box

        stream.stop_stream()
        stream.close()
        audio.terminate()
        pathlib.Path('Speaker/' + string).mkdir(parents=True, exist_ok=True)  # checks if dir exists

        i = 0
        while os.path.exists('Speaker/' + string + '/' + string + '-%s' % i + ".wav"):
            i += 1

        waveFile = wave.open('Speaker/' + string + '/' + string + '-%s' % i + ".wav", 'wb')

        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TheThirdEye"))
        self.pushButtonRecord.setText(_translate("MainWindow", "Record"))
        self.checkBoxRecord.setText(_translate("MainWindow", "Recording"))
        self.labelNameDisplay.setText(_translate("MainWindow", "Your Name:"))
        self.labelAddNewRecording.setText(_translate("MainWindow", "Add New Recording:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Main"))
        self.labelChangeName.setText(_translate("MainWindow", "Change Name for ID:"))
        self.labelAddPeople.setText(_translate("MainWindow", "Add People:"))
        self.pushButtonAddPeople.setText(_translate("MainWindow", "Enter"))
        self.pushButtonChangeName.setText(_translate("MainWindow", "Enter"))
        self.commandLinkButtonNewRecording.setText(_translate("MainWindow", "Record"))
        self.commandLinkButtonLogout.setText(_translate("MainWindow", "Logout"))
        self.commandLinkButtonTrainModel.setText(_translate("commandLinkButtonTrainModel", "Train Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Options"))

Queue = queue.Queue(20)

class Thread1(QtCore.QThread):
    """
    Thread 1 is running in background and records new recordings
    """
    def __init__(self, parent, interval=4):
        QtCore.QThread.__init__(self, parent)

        self.window = parent
        self.interval = interval
        self._lock1 = threading.Lock()
        self.running = False

    def stop(self):
        """
        Stops the thread

        """
        self.running = False
        print('received stop signal from window.')
        with self._lock1:
            self._do_before_done()

    def _do_work(self):
        """
        Records and puts it in queue for the detection

        """
        print('thread is running...')
        count = 0
        global Queue
        record(count)
        Queue.put(count)
        logging.info("Produced " + str(count))
        count = count + 1
        if count == 50:
            count = 0
        time.sleep(self.interval)


    def _do_before_done(self):
        """
        Puts the thread to sleep before closing it to help with scheduling
        """
        self.sleep(1)
        self.sleep(1)

    def run(self):
        self.running = True
        while self.running:
            with self._lock1:
                self._do_work()

class Thread2(QtCore.QThread):
    """
    This thread responsible for the detection keyword and speaker
    """
    def __init__(self, parent):
        QtCore.QThread.__init__(self, parent)

        self.window = parent
        self.flag = False
        self._lock2 = threading.Lock()
        self.running = False

    def stop(self):
        """
        Stops the thread
        """
        self.running = False
        print('received stop signal from window.')
        if self._lock2:
            self._do_before_done()

    def _do_work(self):
        """
        Activated the first model on the recording from the queue
        if there is detection of keyword start the second model to identify the speaker
        :return:
        """
        num = Queue.get()

        model = load_model('model.h5', compile=False)
        model.load_weights('model_weight.h5')

        reader = csv.reader(open('speech.csv', 'r'))
        labels_dict2 = {}
        for row in reader:
            k, v = row
            labels_dict2[k] = v

        file = audio2vector('temp/output' + str(num) + '.wav')
        file = file.reshape(1, 10, 4, 1)

        item_prediction = model.predict_classes(file)
        for i, j in labels_dict2.items():
            if j == str(item_prediction[0]):
                pred = i
                print('keyword: {}'.format(i))
        with open('Files/ID_Name.txt') as fp:
            for line in enumerate(fp):
                if(pred == line[1].strip()):
                    model2 = load_model('model2.h5', compile=False)
                    model2.load_weights('model_weight2.h5')
                    file_name = 'temp/output' + str(num) + '.wav'
                    audio2Image('temp', file_name, 0)
                    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2,
                                                       horizontal_flip=True)
                    training_set = train_datagen.flow_from_directory('./data/test', target_size=(64, 64), batch_size=32,
                                                                     class_mode='categorical', shuffle=False)

                    pred = model2.predict_generator(training_set, steps=10, verbose=1)
                    filenames = training_set.filenames
                    predicted_class_indices = np.argmax(pred, axis=1)
                    labels = (training_set.class_indices)
                    labels = dict((v, k) for k, v in labels.items())
                    predictions = [labels[k] for k in predicted_class_indices]
                    predictions = predictions[:len(filenames)]
                    print('speaker: {}'.format(predictions[0]))
                    #Utils.lower_sound(1)
                    pymsgbox.alert(' {} is calling you.'.format(predictions[0]), 'Recognition')
                    #Utils.lower_sound(0)
                    self.flag=True
        self.sleep(1)



    def _do_before_done(self):
        self.sleep(1)


    def run(self):
        print("x")
        self.running = True
        while self.running:
            with self._lock2:
                self._do_work()




class Register(QtWidgets.QWidget):
    """
    Register page
    """
    switch_window1 = QtCore.pyqtSignal()
    switch_window2 = QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('Register')
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setObjectName("Dialog")
        self.setWindowIcon(QtGui.QIcon('eye-512.png'))
        self.setFixedSize(638, 441)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(110, 190, 151, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(110, 260, 151, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(110, 300, 171, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(110, 230, 161, 31))
        self.label_4.setObjectName("label_4")
        self.txtUsername = QtWidgets.QLineEdit(self)
        self.txtUsername.setGeometry(QtCore.QRect(290, 190, 221, 27))
        self.txtUsername.setObjectName("txtUsername")
        self.txtEmail = QtWidgets.QLineEdit(self)
        self.txtEmail.setGeometry(QtCore.QRect(290, 230, 221, 27))
        self.txtEmail.setObjectName("txtEmail")
        self.txtPassword = QtWidgets.QLineEdit(self)
        ################## make the password invisible ############
        self.txtPassword.setEchoMode(QtWidgets.QLineEdit.Password)
        ###########################################################
        self.txtPassword.setGeometry(QtCore.QRect(290, 270, 221, 27))
        self.txtPassword.setObjectName("txtPassword")
        self.txtPassword2 = QtWidgets.QLineEdit(self)
        ################## make the password2 invisible ############
        self.txtPassword2.setEchoMode(QtWidgets.QLineEdit.Password)
        ###########################################################
        self.txtPassword2.setGeometry(QtCore.QRect(290, 310, 221, 27))
        self.txtPassword2.setObjectName("txtPassword2")

        self.btnRegister = QtWidgets.QPushButton(self)
        self.btnRegister.setGeometry(QtCore.QRect(120, 360, 131, 41))
        self.btnRegister.setObjectName("btnRegister")

        self.btnCloseRegister = QtWidgets.QPushButton(self)
        self.btnCloseRegister.setGeometry(QtCore.QRect(360, 360, 131, 41))
        self.btnCloseRegister.setObjectName("btnCloseRegister")
        ################## register button#########################
        self.btnRegister.clicked.connect(self.registerButton)
        self.btnCloseRegister.clicked.connect(self.closeRegister)
        ###########################################################
        newfont = QtGui.QFont("Aerial", 15, QtGui.QFont.Bold)
        self.label_Heading = QtWidgets.QLabel(self)
        self.label_Heading.setGeometry(QtCore.QRect(105, 30, 431, 61))
        self.label_Heading.setObjectName("label_Heading")
        self.label_Heading.setFont(newfont)
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(110, 150, 151, 31))
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(110, 150, 151, 31))
        self.label_6.setObjectName("label_6")
        self.txtName = QtWidgets.QLineEdit(self)
        self.txtName.setGeometry(QtCore.QRect(290, 150, 221, 27))
        self.txtName.setObjectName("txtName")

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def registerButton(self):
        """
        sends the info to the database and checks if the fields are acceptable

        """
        name = self.txtName.text()
        email = self.txtEmail.text()
        username = self.txtUsername.text()
        password = self.txtPassword.text()
        password2 = self.txtPassword2.text()
        if self.checkFields(username, name, email, password):
            self.showMessage("Error", "All fields must be filled")
        else:
            if (self.checkPassword(password, password2)):
                if(Db().registerCheck(username, email)):
                    Db().insertTable(name, username, email, password)
                    self.showMessage("Success", "Registration successful")
                    self.clearField()
                    self.trainModel()
                else:
                    self.showMessage("Error", "Username or Email already existing")
                    self.clearField()
            else:
                self.showMessage("Error", "Passwords doesn't match")

    def showMessage(self, title, msg):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setWindowIcon(QtGui.QIcon('eye-512.png'))
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.setWindowTitle(title)
        msgBox.setText(msg)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Register"))
        self.label.setText(_translate("Dialog", "UserName:"))
        self.label_2.setText(_translate("Dialog", "Password:"))
        self.label_3.setText(_translate("Dialog", "Repeat Password:"))
        self.label_4.setText(_translate("Dialog", "Email Address:"))
        self.btnRegister.setText(_translate("Dialog", "Register"))
        self.btnCloseRegister.setText(_translate("Dialog", "Close"))
        self.label_Heading.setText(_translate("Dialog", "Create an Account"))

        self.label_6.setText(_translate("Dialog", "First Name:"))



    def checkFields(self, username, name, email, password):
        if (username == "" or name == "" or email == "" or password == ""):
            return True

        ############## check if password1 and password2 matches #############

    def checkPassword(self, password1, password2):
        return password1 == password2

        ##################### clear fields ##################

    def clearField(self):
        self.txtUsername.setText(None)
        self.txtPassword.setText(None)
        self.txtName.setText(None)
        self.txtEmail.setText(None)
        self.txtPassword2.setText(None)

    def closeRegister(self):   #goes back to the login page
        self.switch_window1.emit()

    def trainModel(self):
        """
        goes to train model page

        """
        self.switch_window2.emit()


class Login(QtWidgets.QWidget):
    """
    Login Page
    """
    switch_window1 = QtCore.pyqtSignal()
    switch_window2 = QtCore.pyqtSignal()


    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('Login')
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QtGui.QIcon('eye-512.png'))
        self.resize(500, 120)


        layout = QGridLayout()

        self.label_name = QLabel('<font size="4"> Username </font>')
        self.lineEdit_username = QLineEdit()
        self.lineEdit_username.setPlaceholderText('Please enter your username')
        layout.addWidget(self.label_name, 0, 0)
        layout.addWidget(self.lineEdit_username, 0, 1)

        self.label_password = QLabel('<font size="4"> Password </font>')
        self.lineEdit_password = QLineEdit()
        self.lineEdit_password.setPlaceholderText('Please enter your password')
        layout.addWidget(self.label_password, 1, 0)
        layout.addWidget(self.lineEdit_password, 1, 1)

        self.button_login = QPushButton('Login')
        self.button_login.clicked.connect(self.check_password)
        layout.addWidget(self.button_login, 2, 0, 1, 2)
        layout.setRowMinimumHeight(2, 75)

        self.button_register = QPushButton('Register')
        self.button_register.clicked.connect(self.register)
        layout.addWidget(self.button_register, 3, 0, 2, 2)
        layout.setRowMinimumHeight(2, 75)

        self.setLayout(layout)

    def check_password(self):
        """
        checks the login info
        """
        msg = QMessageBox()
        username = self.lineEdit_username.text()
        password = self.lineEdit_password.text()
        getDb = Db()
        result = getDb.loginCheck(username, password)
        if result:
            msg = QMessageBox()
            msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
            msg.setWindowTitle("Login")
            msg.setText("Success")
            x = msg.exec_()  # this will show our message box
            self.login()

        else:
            msg = QMessageBox()
            msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
            msg.setWindowTitle("Login")
            msg.setText("Incorrect Password")
            x = msg.exec_()  # this will show our message box


    def login(self):   #goes to main page
        self.switch_window1.emit()

    def register(self):      #goes to register page
        self.switch_window2.emit()

class TrainModel(QtWidgets.QWidget):
    """
    Train Model Page
    """
    switch_window = QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle('Train Model')
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QtGui.QIcon('eye-512.png'))
        self.resize(500, 120)

        layout = QGridLayout()

        self.commandLinkButtonNewRecording = QPushButton("Start recording")
        self.commandLinkButtonNewRecording.setGeometry(QtCore.QRect(200, 280, 30, 10))
        self.commandLinkButtonNewRecording.setObjectName("commandLinkButtonNewRecording")

        self.lineEditAddNameTag = QtWidgets.QLineEdit("Please enter your name")
        self.lineEditAddNameTag.setGeometry(QtCore.QRect(240, 245, 120, 30))
        self.lineEditAddNameTag.setObjectName("lineEditAddNameTag")

        button_finish = QPushButton('Finish')
        button_finish.clicked.connect(self.next)
        layout.addWidget(button_finish, 3, 1, 1, 2)
        #layout.setRowMinimumHeight(2, 75)

        layout.addWidget(self.commandLinkButtonNewRecording, 2, 1, 1, 2)
        layout.addWidget(self.lineEditAddNameTag, 1, 1, 1, 2)

        self.commandLinkButtonNewRecording.clicked.connect(self.createRecord)

        self.setLayout(layout)

    def next(self):      #starts the function to train the model

        model = Speech()
        model.expend_data()
        model.read_data()
        model.preprocess_labels()
        model.preprocess_data()
        model.train(10, 4, 1, 1000)

        model2 = Speaker()
        model2.audio_to_image()
        model2.preprocess()
        model2.train()

        self.switch_window.emit()


    def createRecord(self):
        value = self.lineEditAddNameTag.text()
        self.createNewRecording(value)

    def createNewRecording(self, string):

        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
        msg.setWindowTitle("Recording")
        msg.setText("For Training The Model Please record 15 recordings ")
        x = msg.exec_()  # this will show our message box

        for x in range(14):
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 2
            audio = pyaudio.PyAudio()
            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True,
                                frames_per_buffer=CHUNK)
            print("recording...")
            msg = QMessageBox()
            msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
            msg.setWindowTitle("Recording")
            msg.setText("After pressing OK the program will start recording...")
            x = msg.exec_()  # this will show our message box

            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")

            # stop Recording
            msg = QMessageBox()
            msg.setWindowIcon(QtGui.QIcon('eye-512.png'))
            msg.setWindowTitle("Finished")
            msg.setText("Finished Recording")
            x = msg.exec_()  # this will show our message box

            stream.stop_stream()
            stream.close()
            audio.terminate()
            pathlib.Path('Speaker/'+string).mkdir(parents=True, exist_ok=True)  # checks if dir exists

            i = 0
            while os.path.exists('Speaker/' + string + '/' + string + '-%s' % i + ".wav"):
                i += 1

            waveFile = wave.open('Speaker/' + string + '/' + string + '-%s' % i + ".wav", 'wb')

            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

class Controller:
    """
    This class controls all of our switches of pages in the GUI
    """
    def __init__(self):
        pass


    def show_login(self):
        """
        All the switches of login page
        """
        self.register = Register()
        self.window = MainWindow()
        self.login = Login()
        self.login.switch_window1.connect(self.show_main)
        self.login.switch_window2.connect(self.show_register)
        self.login.show()
        self.register.close()
        self.window.close()

    def show_main(self):
        """
        All the switches of main page
        """
        self.window = MainWindow()
        self.train_model = TrainModel()
        self.login = Login()
        self.window.switch_window.connect(self.show_login)
        self.login.close()
        self.train_model.close()
        self.window.show()



    def show_register(self):
        """
        All the switches of register page
        """
        self.register = Register()
        self.register.switch_window1.connect(self.show_login)
        self.register.switch_window2.connect(self.show_train_model)
        self.login.close()
        self.register.show()

    def show_train_model(self):
        """
        All the switches of train model page
        """
        self.register = Register()
        self.window = MainWindow()
        self.train_model = TrainModel()
        self.train_model.switch_window.connect(self.show_main)
        self.train_model.show()
        self.register.close()






def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = Controller()
    controller.show_login()
    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook
    sys.exit(app.exec_())

def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


if __name__ == '__main__':
    main()
