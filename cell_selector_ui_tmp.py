# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cell-selector.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1810, 716)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(300, 290, 1241, 270))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.prev_marked_fame_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.prev_marked_fame_btn.setMaximumSize(QtCore.QSize(285, 16777215))
        self.prev_marked_fame_btn.setObjectName("prev_marked_fame_btn")
        self.horizontalLayout_2.addWidget(self.prev_marked_fame_btn)
        self.prev_frame_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.prev_frame_btn.setMaximumSize(QtCore.QSize(250, 16777215))
        self.prev_frame_btn.setObjectName("prev_frame_btn")
        self.horizontalLayout_2.addWidget(self.prev_frame_btn)
        self.cur_frame_txt = QtWidgets.QTextEdit(self.layoutWidget)
        self.cur_frame_txt.setMaximumSize(QtCore.QSize(90, 90))
        self.cur_frame_txt.setObjectName("cur_frame_txt")
        self.horizontalLayout_2.addWidget(self.cur_frame_txt)
        self.next_frame_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.next_frame_btn.setMaximumSize(QtCore.QSize(250, 16777215))
        self.next_frame_btn.setObjectName("next_frame_btn")
        self.horizontalLayout_2.addWidget(self.next_frame_btn)
        self.next_marked_frame_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.next_marked_frame_btn.setMaximumSize(QtCore.QSize(285, 16777215))
        self.next_marked_frame_btn.setObjectName("next_marked_frame_btn")
        self.horizontalLayout_2.addWidget(self.next_marked_frame_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(62)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.load_vessel_mask_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.load_vessel_mask_btn.setObjectName("load_vessel_mask_btn")
        self.horizontalLayout.addWidget(self.load_vessel_mask_btn)
        self.load_cell_positions_csv_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.load_cell_positions_csv_btn.setObjectName("load_cell_positions_csv_btn")
        self.horizontalLayout.addWidget(self.load_cell_positions_csv_btn)
        self.load_mask_video_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.load_mask_video_btn.setObjectName("load_mask_video_btn")
        self.horizontalLayout.addWidget(self.load_mask_video_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.debugDisplay = QtWidgets.QTextBrowser(self.centralwidget)
        self.debugDisplay.setGeometry(QtCore.QRect(200, 630, 256, 192))
        self.debugDisplay.setObjectName("debugDisplay")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(30, 10, 621, 179))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.mask_video_file_lbl = QtWidgets.QLabel(self.widget)
        self.mask_video_file_lbl.setObjectName("mask_video_file_lbl")
        self.gridLayout.addWidget(self.mask_video_file_lbl, 3, 0, 1, 1)
        self.file_loaded_lbl = QtWidgets.QLabel(self.widget)
        self.file_loaded_lbl.setObjectName("file_loaded_lbl")
        self.gridLayout.addWidget(self.file_loaded_lbl, 0, 0, 1, 1)
        self.cell_positions_csv_lbl = QtWidgets.QLabel(self.widget)
        self.cell_positions_csv_lbl.setObjectName("cell_positions_csv_lbl")
        self.gridLayout.addWidget(self.cell_positions_csv_lbl, 2, 0, 1, 1)
        self.vessel_mask_loaded_lbl = QtWidgets.QLabel(self.widget)
        self.vessel_mask_loaded_lbl.setObjectName("vessel_mask_loaded_lbl")
        self.gridLayout.addWidget(self.vessel_mask_loaded_lbl, 1, 0, 1, 1)
        self.filed_loaded_target_lbl = QtWidgets.QLabel(self.widget)
        self.filed_loaded_target_lbl.setObjectName("filed_loaded_target_lbl")
        self.gridLayout.addWidget(self.filed_loaded_target_lbl, 0, 1, 1, 1)
        self.vessel_mask_loaded_target_lbl = QtWidgets.QLabel(self.widget)
        self.vessel_mask_loaded_target_lbl.setObjectName("vessel_mask_loaded_target_lbl")
        self.gridLayout.addWidget(self.vessel_mask_loaded_target_lbl, 1, 1, 1, 1)
        self.cell_positions_csv_target_lbl = QtWidgets.QLabel(self.widget)
        self.cell_positions_csv_target_lbl.setObjectName("cell_positions_csv_target_lbl")
        self.gridLayout.addWidget(self.cell_positions_csv_target_lbl, 2, 1, 1, 1)
        self.mask_video_file_target_lbl = QtWidgets.QLabel(self.widget)
        self.mask_video_file_target_lbl.setObjectName("mask_video_file_target_lbl")
        self.gridLayout.addWidget(self.mask_video_file_target_lbl, 3, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1810, 47))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.load_vessel_mask_btn.clicked.connect(MainWindow.loadVesselMaskSlot)
        self.load_cell_positions_csv_btn.clicked.connect(MainWindow.loadVesselMaskSlot)
        self.load_mask_video_btn.clicked.connect(MainWindow.LoadMaskVidSlot)
        self.next_frame_btn.clicked.connect(MainWindow.goToNextFrameSlot)
        self.prev_frame_btn.clicked.connect(MainWindow.goToPrevFrameSlot)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.prev_marked_fame_btn.setText(_translate("MainWindow", "<< Prev Marked Frame"))
        self.prev_frame_btn.setText(_translate("MainWindow", "< Prev Frame"))
        self.next_frame_btn.setText(_translate("MainWindow", "Next Frame >"))
        self.next_marked_frame_btn.setText(_translate("MainWindow", "Next Marked Frame >>"))
        self.load_vessel_mask_btn.setText(_translate("MainWindow", "Load Vessel Mask"))
        self.load_cell_positions_csv_btn.setText(_translate("MainWindow", "Load Cell Positions Csv"))
        self.load_mask_video_btn.setText(_translate("MainWindow", "Load Mask vid"))
        self.mask_video_file_lbl.setText(_translate("MainWindow", "Mask Video  file:"))
        self.file_loaded_lbl.setText(_translate("MainWindow", "File loaded:"))
        self.cell_positions_csv_lbl.setText(_translate("MainWindow", "Cell Positions Csv:"))
        self.vessel_mask_loaded_lbl.setText(_translate("MainWindow", "Vessel Mask Loaded:"))
        self.filed_loaded_target_lbl.setText(_translate("MainWindow", "-"))
        self.vessel_mask_loaded_target_lbl.setText(_translate("MainWindow", "TextLabel"))
        self.cell_positions_csv_target_lbl.setText(_translate("MainWindow", "TextLabel"))
        self.mask_video_file_target_lbl.setText(_translate("MainWindow", "TextLabel"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionNew.setText(_translate("MainWindow", "New Video"))
        self.actionNew.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setStatusTip(_translate("MainWindow", "Save current selected cell positions."))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionSave_As.setStatusTip(_translate("MainWindow", "Save file to a new location"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
