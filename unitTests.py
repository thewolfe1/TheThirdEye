import unittest

import FinalGUI
import preprocess
import Utils
from PyQt5 import QtWidgets
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import sys

class TestPreprocess(unittest.TestCase):

    def test_audio2vector(self):
        file='recordings/Alex/alex1.wav'
        item=preprocess.audio2vector(file)
        assert item is not None

    def test_audio2vector2(self):
        file='recordings/Alex/alex1.wav'
        item=preprocess.audio2vector2(file)
        assert item is not None

class TestUtils(unittest.TestCase):

    def test_lower_sound(self):
        item=Utils.lower_sound(True)
        self.assertEqual(item,1)

    def test_lower_sound2(self):
        item = Utils.lower_sound(False)
        self.assertEqual(item, 0)


class TestGUI(unittest.TestCase):
    app = QtWidgets.QApplication(sys.argv)
    register = FinalGUI.Register()
    main_window = FinalGUI.MainWindow()
    # Test Login Page
    def test_loginRegister(self):
        login = FinalGUI.Login()
        login.button_register.setCheckable(True)
        login.button_register.click()
        #QTest.mouseClick(login.button_register, Qt.LeftButton)
        self.assertEqual(login.button_register.isChecked(), True)

    def test_loginButton(self):
        login = FinalGUI.Login()
        login.button_login.setCheckable(True)
        login.button_login.click()
        #QTest.mouseClick(login.button_login, Qt.LeftButton)
        self.assertEqual(login.button_login.isChecked(), True)

    #Register Page Testing
    def test_registerButton(self):
        register = FinalGUI.Register()
        register.btnRegister.setCheckable(True)
        register.btnRegister.click()
        #QTest.mouseClick(register.btnRegister, Qt.LeftButton)
        self.assertEqual(register.btnRegister.isChecked(), True)

    def test_registerBackButton(self):
        register = FinalGUI.Register()
        register.btnCloseRegister.setCheckable(True)
        register.btnCloseRegister.click()
        #QTest.mouseClick(register.btnCloseRegister, Qt.LeftButton)
        self.assertEqual(register.btnCloseRegister.isChecked(), True)

    # Main Page Testing
    def test_RecordButton(self):
        main_window = FinalGUI.MainWindow()
        main_window.pushButtonRecord.setCheckable(True)
        main_window.pushButtonRecord.click()
        #QTest.mouseClick(main_window.pushButtonRecord, Qt.LeftButton)
        self.assertEqual(main_window.pushButtonRecord.isChecked(), True)

    def test_NewRecordingButton(self):
        main_window = FinalGUI.MainWindow()
        main_window.commandLinkButtonNewRecording.setCheckable(True)
        main_window.commandLinkButtonNewRecording.click()
        #QTest.mouseClick(main_window.commandLinkButtonNewRecording, Qt.LeftButton)
        self.assertEqual(main_window.commandLinkButtonNewRecording.isChecked(), True)

    def test_LogoutButton(self):
        main_window = FinalGUI.MainWindow()
        main_window.commandLinkButtonLogout.setCheckable(True)
        main_window.commandLinkButtonLogout.click()
        #QTest.mouseClick(main_window.commandLinkButtonLogout, Qt.LeftButton)
        self.assertEqual(main_window.commandLinkButtonLogout.isChecked(), True)

    def test_ChangeNameButton(self):
        main_window = FinalGUI.MainWindow()
        main_window.pushButtonChangeName.setCheckable(True)
        main_window.pushButtonChangeName.click()
        #QTest.mouseClick(main_window.pushButtonChangeName, Qt.LeftButton)
        self.assertEqual(main_window.pushButtonChangeName.isChecked(), True)

    def test_AddPeopleButton(self):
        main_window = FinalGUI.MainWindow()
        main_window.pushButtonAddPeople.setCheckable(True)
        main_window.pushButtonAddPeople.click()
        #QTest.mouseClick(main_window.pushButtonAddPeople, Qt.LeftButton)
        self.assertEqual(main_window.pushButtonAddPeople.isChecked(), True)


if __name__ == '__main__':
    unittest.main()