# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sample.ui'
#
# Created: Sun Oct  1 15:50:26 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(988, 737)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 60, 169, 31))
        self.label.setMaximumSize(QtCore.QSize(169, 91))
        self.label.setObjectName("label")
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 169, 31))
        self.label_2.setMaximumSize(QtCore.QSize(169, 91))
        self.label_2.setObjectName("label_2")
        self.outputDirLabel = QtGui.QLabel(self.centralwidget)
        self.outputDirLabel.setGeometry(QtCore.QRect(20, 110, 169, 31))
        self.outputDirLabel.setMaximumSize(QtCore.QSize(169, 91))
        self.outputDirLabel.setObjectName("outputDirLabel")
        self.resamplingAlgorithms = QtGui.QComboBox(self.centralwidget)
        self.resamplingAlgorithms.setEnabled(False)
        self.resamplingAlgorithms.setGeometry(QtCore.QRect(200, 60, 211, 27))
        self.resamplingAlgorithms.setObjectName("resamplingAlgorithms")
        self.resamplingAlgorithms.addItem("")
        self.resamplingAlgorithms.addItem("")
        self.resamplingAlgorithms.addItem("")
        self.datasetButton = QtGui.QPushButton(self.centralwidget)
        self.datasetButton.setGeometry(QtCore.QRect(90, 10, 121, 34))
        self.datasetButton.setObjectName("datasetButton")
        self.outputDirectoryButton = QtGui.QPushButton(self.centralwidget)
        self.outputDirectoryButton.setEnabled(False)
        self.outputDirectoryButton.setGeometry(QtCore.QRect(150, 110, 121, 34))
        self.outputDirectoryButton.setObjectName("outputDirectoryButton")
        self.datasetPickedLabel = QtGui.QLabel(self.centralwidget)
        self.datasetPickedLabel.setGeometry(QtCore.QRect(230, 10, 331, 31))
        self.datasetPickedLabel.setObjectName("datasetPickedLabel")
        self.outputDirectoryPickedLabel = QtGui.QLabel(self.centralwidget)
        self.outputDirectoryPickedLabel.setGeometry(QtCore.QRect(280, 110, 331, 31))
        self.outputDirectoryPickedLabel.setObjectName("outputDirectoryPickedLabel")
        self.startButton = QtGui.QPushButton(self.centralwidget)
        self.startButton.setEnabled(False)
        self.startButton.setGeometry(QtCore.QRect(20, 170, 112, 34))
        self.startButton.setObjectName("startButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 988, 31))
        self.menubar.setObjectName("menubar")
        self.menuAsd = QtGui.QMenu(self.menubar)
        self.menuAsd.setObjectName("menuAsd")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionQqq = QtGui.QAction(MainWindow)
        self.actionQqq.setObjectName("actionQqq")
        self.menuAsd.addSeparator()
        self.menuAsd.addSeparator()
        self.menuAsd.addAction(self.actionQqq)
        self.menubar.addAction(self.menuAsd.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Resampling Algorithm:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "Dataset:", None, QtGui.QApplication.UnicodeUTF8))
        self.outputDirLabel.setText(QtGui.QApplication.translate("MainWindow", "Output directory:", None, QtGui.QApplication.UnicodeUTF8))
        self.resamplingAlgorithms.setItemText(0, QtGui.QApplication.translate("MainWindow", "Random Oversampling", None, QtGui.QApplication.UnicodeUTF8))
        self.resamplingAlgorithms.setItemText(1, QtGui.QApplication.translate("MainWindow", "Random Undersampling", None, QtGui.QApplication.UnicodeUTF8))
        self.resamplingAlgorithms.setItemText(2, QtGui.QApplication.translate("MainWindow", "Smote", None, QtGui.QApplication.UnicodeUTF8))
        self.datasetButton.setText(QtGui.QApplication.translate("MainWindow", "Choose...", None, QtGui.QApplication.UnicodeUTF8))
        self.outputDirectoryButton.setText(QtGui.QApplication.translate("MainWindow", "Choose...", None, QtGui.QApplication.UnicodeUTF8))
        self.datasetPickedLabel.setText(QtGui.QApplication.translate("MainWindow", "Empty", None, QtGui.QApplication.UnicodeUTF8))
        self.outputDirectoryPickedLabel.setText(QtGui.QApplication.translate("MainWindow", "Empty", None, QtGui.QApplication.UnicodeUTF8))
        self.startButton.setText(QtGui.QApplication.translate("MainWindow", "Start", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAsd.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQqq.setText(QtGui.QApplication.translate("MainWindow", "Test", None, QtGui.QApplication.UnicodeUTF8))

