# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rotation.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(848, 1298)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.four_way = QtWidgets.QWidget(Form)
        self.four_way.setObjectName("four_way")
        self.gridLayout = QtWidgets.QGridLayout(self.four_way)
        self.gridLayout.setObjectName("gridLayout")
        self.Euler = QtWidgets.QGroupBox(self.four_way)
        self.Euler.setObjectName("Euler")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.Euler)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.widget_5 = QtWidgets.QWidget(self.Euler)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.groupBox_14 = QtWidgets.QGroupBox(self.widget_5)
        self.groupBox_14.setEnabled(False)
        self.groupBox_14.setObjectName("groupBox_14")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.groupBox_14)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.alpha = QtWidgets.QDoubleSpinBox(self.groupBox_14)
        self.alpha.setMinimum(-10.0)
        self.alpha.setMaximum(10.0)
        self.alpha.setProperty("value", 1.0)
        self.alpha.setObjectName("alpha")
        self.horizontalLayout_13.addWidget(self.alpha)
        self.horizontalLayout_12.addWidget(self.groupBox_14)
        self.groupBox_15 = QtWidgets.QGroupBox(self.widget_5)
        self.groupBox_15.setEnabled(False)
        self.groupBox_15.setObjectName("groupBox_15")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.groupBox_15)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.beta = QtWidgets.QDoubleSpinBox(self.groupBox_15)
        self.beta.setMinimum(-10.0)
        self.beta.setMaximum(10.0)
        self.beta.setObjectName("beta")
        self.horizontalLayout_14.addWidget(self.beta)
        self.horizontalLayout_12.addWidget(self.groupBox_15)
        self.groupBox_16 = QtWidgets.QGroupBox(self.widget_5)
        self.groupBox_16.setEnabled(False)
        self.groupBox_16.setObjectName("groupBox_16")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.groupBox_16)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.gamma = QtWidgets.QDoubleSpinBox(self.groupBox_16)
        self.gamma.setMinimum(-10.0)
        self.gamma.setMaximum(10.0)
        self.gamma.setObjectName("gamma")
        self.horizontalLayout_15.addWidget(self.gamma)
        self.horizontalLayout_12.addWidget(self.groupBox_16)
        self.verticalLayout_5.addWidget(self.widget_5)
        self.Euler_botton = QtWidgets.QPushButton(self.Euler)
        self.Euler_botton.setEnabled(False)
        self.Euler_botton.setObjectName("Euler_botton")
        self.verticalLayout_5.addWidget(self.Euler_botton)
        self.gridLayout.addWidget(self.Euler, 1, 0, 1, 1)
        self.quat = QtWidgets.QGroupBox(self.four_way)
        self.quat.setObjectName("quat")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.quat)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_4 = QtWidgets.QWidget(self.quat)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.groupBox_9 = QtWidgets.QGroupBox(self.widget_4)
        self.groupBox_9.setObjectName("groupBox_9")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.groupBox_9)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.quatw = QtWidgets.QDoubleSpinBox(self.groupBox_9)
        self.quatw.setMinimum(-100.0)
        self.quatw.setMaximum(100.0)
        self.quatw.setProperty("value", 1.0)
        self.quatw.setObjectName("quatw")
        self.horizontalLayout_8.addWidget(self.quatw)
        self.horizontalLayout_7.addWidget(self.groupBox_9)
        self.groupBox_10 = QtWidgets.QGroupBox(self.widget_4)
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.groupBox_10)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.quatx = QtWidgets.QDoubleSpinBox(self.groupBox_10)
        self.quatx.setMinimum(-100.0)
        self.quatx.setMaximum(100.0)
        self.quatx.setObjectName("quatx")
        self.horizontalLayout_9.addWidget(self.quatx)
        self.horizontalLayout_7.addWidget(self.groupBox_10)
        self.groupBox_11 = QtWidgets.QGroupBox(self.widget_4)
        self.groupBox_11.setObjectName("groupBox_11")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox_11)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.quaty = QtWidgets.QDoubleSpinBox(self.groupBox_11)
        self.quaty.setMinimum(-100.0)
        self.quaty.setMaximum(100.0)
        self.quaty.setObjectName("quaty")
        self.horizontalLayout_10.addWidget(self.quaty)
        self.horizontalLayout_7.addWidget(self.groupBox_11)
        self.groupBox_12 = QtWidgets.QGroupBox(self.widget_4)
        self.groupBox_12.setObjectName("groupBox_12")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.groupBox_12)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.quatz = QtWidgets.QDoubleSpinBox(self.groupBox_12)
        self.quatz.setMinimum(-100.0)
        self.quatz.setMaximum(100.0)
        self.quatz.setObjectName("quatz")
        self.horizontalLayout_11.addWidget(self.quatz)
        self.horizontalLayout_7.addWidget(self.groupBox_12)
        self.verticalLayout_2.addWidget(self.widget_4)
        self.groupBox_13 = QtWidgets.QGroupBox(self.quat)
        self.groupBox_13.setObjectName("groupBox_13")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_13)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_3 = QtWidgets.QWidget(self.groupBox_13)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.groupBox_5 = QtWidgets.QGroupBox(self.widget_3)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.unitw = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.unitw.setMinimum(-1.0)
        self.unitw.setMaximum(1.0)
        self.unitw.setProperty("value", 1.0)
        self.unitw.setObjectName("unitw")
        self.horizontalLayout_5.addWidget(self.unitw)
        self.horizontalLayout_6.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.widget_3)
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.unitx = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.unitx.setMinimum(-1.0)
        self.unitx.setMaximum(1.0)
        self.unitx.setObjectName("unitx")
        self.horizontalLayout_4.addWidget(self.unitx)
        self.horizontalLayout_6.addWidget(self.groupBox_6)
        self.groupBox_8 = QtWidgets.QGroupBox(self.widget_3)
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_8)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.unity = QtWidgets.QDoubleSpinBox(self.groupBox_8)
        self.unity.setMinimum(-1.0)
        self.unity.setMaximum(1.0)
        self.unity.setObjectName("unity")
        self.horizontalLayout.addWidget(self.unity)
        self.horizontalLayout_6.addWidget(self.groupBox_8)
        self.groupBox_7 = QtWidgets.QGroupBox(self.widget_3)
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.unitz = QtWidgets.QDoubleSpinBox(self.groupBox_7)
        self.unitz.setMinimum(-1.0)
        self.unitz.setMaximum(1.0)
        self.unitz.setObjectName("unitz")
        self.horizontalLayout_3.addWidget(self.unitz)
        self.horizontalLayout_6.addWidget(self.groupBox_7)
        self.horizontalLayout_2.addWidget(self.widget_3)
        self.verticalLayout_2.addWidget(self.groupBox_13)
        self.quat_botton = QtWidgets.QPushButton(self.quat)
        self.quat_botton.setObjectName("quat_botton")
        self.verticalLayout_2.addWidget(self.quat_botton)
        self.gridLayout.addWidget(self.quat, 0, 1, 1, 1)
        self.matrix = QtWidgets.QGroupBox(self.four_way)
        self.matrix.setObjectName("matrix")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.matrix)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widget = QtWidgets.QWidget(self.matrix)
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.matrix20 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix20.setMinimum(-1.0)
        self.matrix20.setMaximum(1.0)
        self.matrix20.setObjectName("matrix20")
        self.gridLayout_2.addWidget(self.matrix20, 2, 0, 1, 1)
        self.matrix00 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix00.setMinimum(-1.0)
        self.matrix00.setMaximum(1.0)
        self.matrix00.setProperty("value", 1.0)
        self.matrix00.setObjectName("matrix00")
        self.gridLayout_2.addWidget(self.matrix00, 0, 0, 1, 1)
        self.matrix01 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix01.setMinimum(-1.0)
        self.matrix01.setMaximum(1.0)
        self.matrix01.setObjectName("matrix01")
        self.gridLayout_2.addWidget(self.matrix01, 0, 1, 1, 1)
        self.matrix11 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix11.setMinimum(-1.0)
        self.matrix11.setMaximum(1.0)
        self.matrix11.setProperty("value", 1.0)
        self.matrix11.setObjectName("matrix11")
        self.gridLayout_2.addWidget(self.matrix11, 1, 1, 1, 1)
        self.matrix21 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix21.setMinimum(-1.0)
        self.matrix21.setMaximum(1.0)
        self.matrix21.setObjectName("matrix21")
        self.gridLayout_2.addWidget(self.matrix21, 2, 1, 1, 1)
        self.matrix10 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix10.setMinimum(-1.0)
        self.matrix10.setMaximum(1.0)
        self.matrix10.setObjectName("matrix10")
        self.gridLayout_2.addWidget(self.matrix10, 1, 0, 1, 1)
        self.matrix02 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix02.setMinimum(-1.0)
        self.matrix02.setMaximum(1.0)
        self.matrix02.setObjectName("matrix02")
        self.gridLayout_2.addWidget(self.matrix02, 0, 2, 1, 1)
        self.matrix22 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix22.setMinimum(-1.0)
        self.matrix22.setMaximum(1.0)
        self.matrix22.setProperty("value", 1.0)
        self.matrix22.setObjectName("matrix22")
        self.gridLayout_2.addWidget(self.matrix22, 2, 2, 1, 1)
        self.matrix12 = QtWidgets.QDoubleSpinBox(self.widget)
        self.matrix12.setMinimum(-1.0)
        self.matrix12.setMaximum(1.0)
        self.matrix12.setObjectName("matrix12")
        self.gridLayout_2.addWidget(self.matrix12, 1, 2, 1, 1)
        self.verticalLayout_6.addWidget(self.widget)
        self.matrix_button = QtWidgets.QPushButton(self.matrix)
        self.matrix_button.setEnabled(False)
        self.matrix_button.setObjectName("matrix_button")
        self.verticalLayout_6.addWidget(self.matrix_button)
        self.gridLayout.addWidget(self.matrix, 0, 0, 1, 1)
        self.angleaxis = QtWidgets.QGroupBox(self.four_way)
        self.angleaxis.setObjectName("angleaxis")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.angleaxis)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_6 = QtWidgets.QWidget(self.angleaxis)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.groupBox_20 = QtWidgets.QGroupBox(self.widget_6)
        self.groupBox_20.setObjectName("groupBox_20")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.groupBox_20)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.angle = QtWidgets.QDoubleSpinBox(self.groupBox_20)
        self.angle.setMinimum(-10.0)
        self.angle.setMaximum(10.0)
        self.angle.setProperty("value", 1.0)
        self.angle.setObjectName("angle")
        self.horizontalLayout_22.addWidget(self.angle)
        self.horizontalLayout_16.addWidget(self.groupBox_20)
        self.line = QtWidgets.QFrame(self.widget_6)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_16.addWidget(self.line)
        self.groupBox_17 = QtWidgets.QGroupBox(self.widget_6)
        self.groupBox_17.setObjectName("groupBox_17")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.groupBox_17)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.vectorx = QtWidgets.QDoubleSpinBox(self.groupBox_17)
        self.vectorx.setMinimum(-10.0)
        self.vectorx.setMaximum(10.0)
        self.vectorx.setProperty("value", 1.0)
        self.vectorx.setObjectName("vectorx")
        self.horizontalLayout_17.addWidget(self.vectorx)
        self.horizontalLayout_16.addWidget(self.groupBox_17)
        self.groupBox_18 = QtWidgets.QGroupBox(self.widget_6)
        self.groupBox_18.setObjectName("groupBox_18")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.groupBox_18)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.vectory = QtWidgets.QDoubleSpinBox(self.groupBox_18)
        self.vectory.setMinimum(-10.0)
        self.vectory.setMaximum(10.0)
        self.vectory.setObjectName("vectory")
        self.horizontalLayout_18.addWidget(self.vectory)
        self.horizontalLayout_16.addWidget(self.groupBox_18)
        self.groupBox_19 = QtWidgets.QGroupBox(self.widget_6)
        self.groupBox_19.setObjectName("groupBox_19")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.groupBox_19)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.vectorz = QtWidgets.QDoubleSpinBox(self.groupBox_19)
        self.vectorz.setMinimum(-10.0)
        self.vectorz.setMaximum(10.0)
        self.vectorz.setObjectName("vectorz")
        self.horizontalLayout_19.addWidget(self.vectorz)
        self.horizontalLayout_16.addWidget(self.groupBox_19)
        self.verticalLayout_3.addWidget(self.widget_6)
        self.angleaxis_botton = QtWidgets.QPushButton(self.angleaxis)
        self.angleaxis_botton.setObjectName("angleaxis_botton")
        self.verticalLayout_3.addWidget(self.angleaxis_botton)
        self.gridLayout.addWidget(self.angleaxis, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.four_way)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.widget_7 = QtWidgets.QWidget(Form)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.textBrowser = QtWidgets.QTextBrowser(self.widget_7)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout_20.addWidget(self.textBrowser)
        self.verticalLayout_4.addWidget(self.widget_7)
        self.widget_2 = QtWidgets.QWidget(Form)
        self.widget_2.setMinimumSize(QtCore.QSize(200, 300))
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_4.addWidget(self.widget_2)

        self.retranslateUi(Form)
        self.matrix_button.clicked.connect(Form.slot_matrix2all)
        self.quat_botton.clicked.connect(Form.slot_quat2all)
        self.angleaxis_botton.clicked.connect(Form.slot_anglevec2all)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Rotation"))
        self.Euler.setTitle(_translate("Form", "欧拉角"))
        self.groupBox_14.setTitle(_translate("Form", "alpha"))
        self.groupBox_15.setTitle(_translate("Form", "beta"))
        self.groupBox_16.setTitle(_translate("Form", "gamma"))
        self.Euler_botton.setText(_translate("Form", "转换"))
        self.quat.setTitle(_translate("Form", "四元数"))
        self.groupBox_9.setTitle(_translate("Form", "w"))
        self.groupBox_10.setTitle(_translate("Form", "x"))
        self.groupBox_11.setTitle(_translate("Form", "y"))
        self.groupBox_12.setTitle(_translate("Form", "z"))
        self.groupBox_13.setTitle(_translate("Form", "单位四元数"))
        self.groupBox_5.setTitle(_translate("Form", "w"))
        self.groupBox_6.setTitle(_translate("Form", "x"))
        self.groupBox_8.setTitle(_translate("Form", "y"))
        self.groupBox_7.setTitle(_translate("Form", "z"))
        self.quat_botton.setText(_translate("Form", "转换"))
        self.matrix.setTitle(_translate("Form", "矩阵"))
        self.matrix_button.setText(_translate("Form", "转换"))
        self.angleaxis.setTitle(_translate("Form", "轴角"))
        self.groupBox_20.setTitle(_translate("Form", "angle"))
        self.groupBox_17.setTitle(_translate("Form", "x"))
        self.groupBox_18.setTitle(_translate("Form", "y"))
        self.groupBox_19.setTitle(_translate("Form", "z"))
        self.angleaxis_botton.setText(_translate("Form", "转换"))
