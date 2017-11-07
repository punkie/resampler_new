# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(988, 737)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 60, 169, 31))
        self.label.setMaximumSize(QtCore.QSize(169, 91))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 169, 31))
        self.label_2.setMaximumSize(QtCore.QSize(169, 91))
        self.label_2.setObjectName("label_2")
        self.outputDirLabel = QtWidgets.QLabel(self.centralwidget)
        self.outputDirLabel.setGeometry(QtCore.QRect(20, 110, 169, 31))
        self.outputDirLabel.setMaximumSize(QtCore.QSize(169, 91))
        self.outputDirLabel.setObjectName("outputDirLabel")
        self.resamplingAlgorithms = QtWidgets.QComboBox(self.centralwidget)
        self.resamplingAlgorithms.setEnabled(False)
        self.resamplingAlgorithms.setGeometry(QtCore.QRect(200, 60, 211, 31))
        self.resamplingAlgorithms.setObjectName("resamplingAlgorithms")
        self.datasetButton = QtWidgets.QPushButton(self.centralwidget)
        self.datasetButton.setGeometry(QtCore.QRect(200, 10, 121, 34))
        self.datasetButton.setObjectName("datasetButton")
        self.outputDirectoryButton = QtWidgets.QPushButton(self.centralwidget)
        self.outputDirectoryButton.setEnabled(False)
        self.outputDirectoryButton.setGeometry(QtCore.QRect(200, 110, 121, 34))
        self.outputDirectoryButton.setObjectName("outputDirectoryButton")
        self.datasetStatisticsLabel = QtWidgets.QLabel(self.centralwidget)
        self.datasetStatisticsLabel.setGeometry(QtCore.QRect(750, 60, 191, 71))
        self.datasetStatisticsLabel.setText("")
        self.datasetStatisticsLabel.setWordWrap(True)
        self.datasetStatisticsLabel.setObjectName("datasetStatisticsLabel")
        self.outputDirectoryPickedLabel = QtWidgets.QLabel(self.centralwidget)
        self.outputDirectoryPickedLabel.setGeometry(QtCore.QRect(440, 110, 261, 51))
        self.outputDirectoryPickedLabel.setText("")
        self.outputDirectoryPickedLabel.setWordWrap(True)
        self.outputDirectoryPickedLabel.setObjectName("outputDirectoryPickedLabel")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setEnabled(False)
        self.startButton.setGeometry(QtCore.QRect(20, 170, 112, 34))
        self.startButton.setObjectName("startButton")
        self.datasetProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.datasetProgressBar.setGeometry(QtCore.QRect(330, 12, 101, 31))
        self.datasetProgressBar.setMaximum(336)
        self.datasetProgressBar.setProperty("value", 0)
        self.datasetProgressBar.setObjectName("datasetProgressBar")
        self.resamplingStatusLabel = QtWidgets.QLabel(self.centralwidget)
        self.resamplingStatusLabel.setGeometry(QtCore.QRect(150, 170, 169, 31))
        self.resamplingStatusLabel.setMaximumSize(QtCore.QSize(169, 91))
        self.resamplingStatusLabel.setText("")
        self.resamplingStatusLabel.setObjectName("resamplingStatusLabel")
        self.datasetPickedLabel = QtWidgets.QLabel(self.centralwidget)
        self.datasetPickedLabel.setGeometry(QtCore.QRect(440, 10, 261, 41))
        self.datasetPickedLabel.setText("")
        self.datasetPickedLabel.setWordWrap(True)
        self.datasetPickedLabel.setObjectName("datasetPickedLabel")
        self.resampledDatasetStatistics = QtWidgets.QLabel(self.centralwidget)
        self.resampledDatasetStatistics.setGeometry(QtCore.QRect(750, 170, 191, 71))
        self.resampledDatasetStatistics.setText("")
        self.resampledDatasetStatistics.setWordWrap(True)
        self.resampledDatasetStatistics.setObjectName("resampledDatasetStatistics")
        self.imgDiffsButton = QtWidgets.QPushButton(self.centralwidget)
        self.imgDiffsButton.setEnabled(False)
        self.imgDiffsButton.setGeometry(QtCore.QRect(790, 260, 121, 34))
        self.imgDiffsButton.setObjectName("imgDiffsButton")
        self.classifyButton = QtWidgets.QPushButton(self.centralwidget)
        self.classifyButton.setEnabled(False)
        self.classifyButton.setGeometry(QtCore.QRect(20, 290, 112, 34))
        self.classifyButton.setObjectName("classifyButton")
        self.classifyLabel = QtWidgets.QLabel(self.centralwidget)
        self.classifyLabel.setGeometry(QtCore.QRect(20, 230, 400, 51))
        self.classifyLabel.setMaximumSize(QtCore.QSize(400, 91))
        self.classifyLabel.setToolTip("")
        self.classifyLabel.setWordWrap(True)
        self.classifyLabel.setObjectName("classifyLabel")
        self.classifyingStatusLabel = QtWidgets.QLabel(self.centralwidget)
        self.classifyingStatusLabel.setGeometry(QtCore.QRect(150, 290, 169, 31))
        self.classifyingStatusLabel.setMaximumSize(QtCore.QSize(169, 91))
        self.classifyingStatusLabel.setToolTip("")
        self.classifyingStatusLabel.setText("")
        self.classifyingStatusLabel.setObjectName("classifyingStatusLabel")
        self.normalClassifyProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.normalClassifyProgressBar.setEnabled(True)
        self.normalClassifyProgressBar.setGeometry(QtCore.QRect(20, 370, 101, 31))
        self.normalClassifyProgressBar.setMaximum(10)
        self.normalClassifyProgressBar.setProperty("value", 0)
        self.normalClassifyProgressBar.setObjectName("normalClassifyProgressBar")
        self.resampleClassifyProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.resampleClassifyProgressBar.setEnabled(True)
        self.resampleClassifyProgressBar.setGeometry(QtCore.QRect(230, 370, 101, 31))
        self.resampleClassifyProgressBar.setMaximum(10)
        self.resampleClassifyProgressBar.setProperty("value", 0)
        self.resampleClassifyProgressBar.setObjectName("resampleClassifyProgressBar")
        self.classifyLabelNormalDataset = QtWidgets.QLabel(self.centralwidget)
        self.classifyLabelNormalDataset.setGeometry(QtCore.QRect(20, 340, 180, 31))
        self.classifyLabelNormalDataset.setMaximumSize(QtCore.QSize(180, 91))
        self.classifyLabelNormalDataset.setToolTip("")
        self.classifyLabelNormalDataset.setObjectName("classifyLabelNormalDataset")
        self.classifyLabelResampledDataset = QtWidgets.QLabel(self.centralwidget)
        self.classifyLabelResampledDataset.setGeometry(QtCore.QRect(230, 340, 200, 31))
        self.classifyLabelResampledDataset.setMaximumSize(QtCore.QSize(200, 91))
        self.classifyLabelResampledDataset.setToolTip("")
        self.classifyLabelResampledDataset.setObjectName("classifyLabelResampledDataset")
        self.showROCGraphs = QtWidgets.QPushButton(self.centralwidget)
        self.showROCGraphs.setEnabled(False)
        self.showROCGraphs.setGeometry(QtCore.QRect(730, 340, 112, 34))
        self.showROCGraphs.setObjectName("showROCGraphs")
        self.showPRGraphs = QtWidgets.QPushButton(self.centralwidget)
        self.showPRGraphs.setEnabled(False)
        self.showPRGraphs.setGeometry(QtCore.QRect(860, 340, 112, 34))
        self.showPRGraphs.setObjectName("showPRGraphs")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(720, 10, 3, 691))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.statsLabelNormalDs = QtWidgets.QLabel(self.centralwidget)
        self.statsLabelNormalDs.setGeometry(QtCore.QRect(750, 20, 200, 31))
        self.statsLabelNormalDs.setMaximumSize(QtCore.QSize(200, 91))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.statsLabelNormalDs.setFont(font)
        self.statsLabelNormalDs.setToolTip("")
        self.statsLabelNormalDs.setObjectName("statsLabelNormalDs")
        self.statsLabelNormalDs_2 = QtWidgets.QLabel(self.centralwidget)
        self.statsLabelNormalDs_2.setGeometry(QtCore.QRect(750, 140, 220, 31))
        self.statsLabelNormalDs_2.setMaximumSize(QtCore.QSize(220, 91))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.statsLabelNormalDs_2.setFont(font)
        self.statsLabelNormalDs_2.setToolTip("")
        self.statsLabelNormalDs_2.setObjectName("statsLabelNormalDs_2")
        self.statsLabelNormalDs_3 = QtWidgets.QLabel(self.centralwidget)
        self.statsLabelNormalDs_3.setGeometry(QtCore.QRect(750, 390, 220, 31))
        self.statsLabelNormalDs_3.setMaximumSize(QtCore.QSize(220, 91))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.statsLabelNormalDs_3.setFont(font)
        self.statsLabelNormalDs_3.setToolTip("")
        self.statsLabelNormalDs_3.setObjectName("statsLabelNormalDs_3")
        self.afterClassificationStatistics = QtWidgets.QLabel(self.centralwidget)
        self.afterClassificationStatistics.setGeometry(QtCore.QRect(740, 430, 231, 161))
        self.afterClassificationStatistics.setText("")
        self.afterClassificationStatistics.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.afterClassificationStatistics.setWordWrap(True)
        self.afterClassificationStatistics.setObjectName("afterClassificationStatistics")
        self.classifyLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.classifyLabel_2.setGeometry(QtCore.QRect(750, 510, 211, 51))
        self.classifyLabel_2.setMaximumSize(QtCore.QSize(400, 91))
        self.classifyLabel_2.setToolTip("")
        self.classifyLabel_2.setWordWrap(True)
        self.classifyLabel_2.setObjectName("classifyLabel_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 988, 21))
        self.menubar.setObjectName("menubar")
        self.menuAsd = QtWidgets.QMenu(self.menubar)
        self.menuAsd.setObjectName("menuAsd")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionQqq = QtWidgets.QAction(MainWindow)
        self.actionQqq.setObjectName("actionQqq")
        self.menuAsd.addSeparator()
        self.menuAsd.addSeparator()
        self.menuAsd.addAction(self.actionQqq)
        self.menubar.addAction(self.menuAsd.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Resampling Algorithm:"))
        self.label_2.setToolTip(_translate("MainWindow", "Only csv datasets are acceptable!"))
        self.label_2.setText(_translate("MainWindow", "Dataset:"))
        self.outputDirLabel.setText(_translate("MainWindow", "Output directory for resampled DS:"))
        self.datasetButton.setToolTip(_translate("MainWindow", "Only csv datasets are acceptable!"))
        self.datasetButton.setText(_translate("MainWindow", "Choose..."))
        self.outputDirectoryButton.setToolTip(_translate("MainWindow", "Choose the output dir, where the resampled dataset file will be stored!"))
        self.outputDirectoryButton.setText(_translate("MainWindow", "Choose..."))
        self.startButton.setToolTip(_translate("MainWindow", "Needs the output dir to be chosen!"))
        self.startButton.setText(_translate("MainWindow", "Start resampling"))
        self.resamplingStatusLabel.setToolTip(_translate("MainWindow", "Only csv datasets are acceptable!"))
        self.imgDiffsButton.setToolTip(_translate("MainWindow", "Needs the resampling done!"))
        self.imgDiffsButton.setText(_translate("MainWindow", "Dataset diffs as imgs"))
        self.classifyButton.setToolTip(_translate("MainWindow", "Needs the dataset to be loaded!"))
        self.classifyButton.setText(_translate("MainWindow", "Classify"))
        self.classifyLabel.setText(_translate("MainWindow", "The button below is doing a 10-Fold Cross-validation using CART on the normal and resampled datasets!"))
        self.classifyLabelNormalDataset.setText(_translate("MainWindow", "Classify progress on normal dataset:"))
        self.classifyLabelResampledDataset.setText(_translate("MainWindow", "Classify progress on resampled dataset:"))
        self.showROCGraphs.setToolTip(_translate("MainWindow", "Needs the classification done!"))
        self.showROCGraphs.setText(_translate("MainWindow", "Show ROC Graphs"))
        self.showPRGraphs.setToolTip(_translate("MainWindow", "Needs the classification done!"))
        self.showPRGraphs.setText(_translate("MainWindow", "Show PR Graphs"))
        self.statsLabelNormalDs.setText(_translate("MainWindow", "Statistics for the standart dataset:"))
        self.statsLabelNormalDs_2.setText(_translate("MainWindow", "Statistics for the resampled dataset:"))
        self.statsLabelNormalDs_3.setText(_translate("MainWindow", "Mean statistics for the two variants:"))
        self.classifyLabel_2.setText(_translate("MainWindow", "The statistics are for the minor class and are averaged for all folds!"))
        self.menuAsd.setTitle(_translate("MainWindow", "File"))
        self.actionQqq.setText(_translate("MainWindow", "Test"))

