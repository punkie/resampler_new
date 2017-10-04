# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sample.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(988, 737)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 60, 169, 31))
        self.label.setMaximumSize(QtCore.QSize(169, 91))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 169, 31))
        self.label_2.setMaximumSize(QtCore.QSize(169, 91))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.outputDirLabel = QtGui.QLabel(self.centralwidget)
        self.outputDirLabel.setGeometry(QtCore.QRect(20, 110, 169, 31))
        self.outputDirLabel.setMaximumSize(QtCore.QSize(169, 91))
        self.outputDirLabel.setObjectName(_fromUtf8("outputDirLabel"))
        self.resamplingAlgorithms = QtGui.QComboBox(self.centralwidget)
        self.resamplingAlgorithms.setEnabled(False)
        self.resamplingAlgorithms.setGeometry(QtCore.QRect(200, 60, 211, 31))
        self.resamplingAlgorithms.setObjectName(_fromUtf8("resamplingAlgorithms"))
        self.resamplingAlgorithms.addItem(_fromUtf8(""))
        self.resamplingAlgorithms.addItem(_fromUtf8(""))
        self.resamplingAlgorithms.addItem(_fromUtf8(""))
        self.datasetButton = QtGui.QPushButton(self.centralwidget)
        self.datasetButton.setGeometry(QtCore.QRect(90, 10, 121, 34))
        self.datasetButton.setObjectName(_fromUtf8("datasetButton"))
        self.outputDirectoryButton = QtGui.QPushButton(self.centralwidget)
        self.outputDirectoryButton.setEnabled(False)
        self.outputDirectoryButton.setGeometry(QtCore.QRect(150, 110, 121, 34))
        self.outputDirectoryButton.setObjectName(_fromUtf8("outputDirectoryButton"))
        self.datasetPickedLabel = QtGui.QLabel(self.centralwidget)
        self.datasetPickedLabel.setGeometry(QtCore.QRect(230, 10, 331, 31))
        self.datasetPickedLabel.setObjectName(_fromUtf8("datasetPickedLabel"))
        self.outputDirectoryPickedLabel = QtGui.QLabel(self.centralwidget)
        self.outputDirectoryPickedLabel.setGeometry(QtCore.QRect(280, 110, 331, 31))
        self.outputDirectoryPickedLabel.setObjectName(_fromUtf8("outputDirectoryPickedLabel"))
        self.startButton = QtGui.QPushButton(self.centralwidget)
        self.startButton.setEnabled(False)
        self.startButton.setGeometry(QtCore.QRect(20, 170, 112, 34))
        self.startButton.setObjectName(_fromUtf8("startButton"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 988, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuAsd = QtGui.QMenu(self.menubar)
        self.menuAsd.setObjectName(_fromUtf8("menuAsd"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionQqq = QtGui.QAction(MainWindow)
        self.actionQqq.setObjectName(_fromUtf8("actionQqq"))
        self.menuAsd.addSeparator()
        self.menuAsd.addSeparator()
        self.menuAsd.addAction(self.actionQqq)
        self.menubar.addAction(self.menuAsd.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Resampling Algorithm:", None))
        self.label_2.setText(_translate("MainWindow", "Dataset:", None))
        self.outputDirLabel.setText(_translate("MainWindow", "Output directory:", None))
        self.resamplingAlgorithms.setItemText(0, _translate("MainWindow", "Random Oversampling", None))
        self.resamplingAlgorithms.setItemText(1, _translate("MainWindow", "Random Undersampling", None))
        self.resamplingAlgorithms.setItemText(2, _translate("MainWindow", "Smote", None))
        self.datasetButton.setText(_translate("MainWindow", "Choose...", None))
        self.outputDirectoryButton.setText(_translate("MainWindow", "Choose...", None))
        self.datasetPickedLabel.setText(_translate("MainWindow", "Empty", None))
        self.outputDirectoryPickedLabel.setText(_translate("MainWindow", "Empty", None))
        self.startButton.setText(_translate("MainWindow", "Start", None))
        self.menuAsd.setTitle(_translate("MainWindow", "File", None))
        self.actionQqq.setText(_translate("MainWindow", "Test", None))

