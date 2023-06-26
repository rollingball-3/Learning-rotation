import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from SO3 import SO3

from rotation import Ui_Form


class My_window(QWidget, Ui_Form):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)
        self.retranslateUi(self)

        self.current_rotation = SO3()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout = QVBoxLayout(self.widget_2)
        self.layout.addWidget(self.canvas)

        self.display_current_rotation()

    def slot_matrix2all(self):
        pass

    def slot_quat2all(self):
        q = np.empty(4)
        q[0] = self.quatw.value()
        q[1] = self.quatx.value()
        q[2] = self.quaty.value()
        q[3] = self.quatz.value()

        self.current_rotation = SO3.from_quaternion(q)

        self.display_current_rotation()

    def slot_anglevec2all(self):
        vector = np.empty(4)
        vector[0] = self.vectorx.value()
        vector[1] = self.vectory.value()
        vector[2] = self.vectorz.value()
        vector[3] = self.angle.value()
        self.current_rotation = SO3.from_axis_angle(vector)

        self.display_current_rotation()

    def display_current_rotation(self):
        # matrix
        matrix = self.current_rotation.rotation_matrix
        print(matrix)
        for i in range(3):
            for j in range(3):
                exec("self.matrix{}{}.setValue({})".format(str(i), str(j), matrix[i][j]))

        # quat
        quat = self.current_rotation.to_quaternion()
        self.quatw.setValue(quat[0])
        self.unitw.setValue(quat[0])
        self.quatx.setValue(quat[1])
        self.unitx.setValue(quat[1])
        self.quaty.setValue(quat[2])
        self.unity.setValue(quat[2])
        self.quatz.setValue(quat[3])
        self.unitz.setValue(quat[3])

        # angle_vector
        vector = self.current_rotation.to_axis_angle()
        self.angle.setValue(vector[3])
        self.vectorx.setValue(vector[0])
        self.vectory.setValue(vector[1])
        self.vectorz.setValue(vector[2])

        self.textBrowser.setText(str(self.current_rotation))

        # plot
        self.figure.clear()
        self.current_rotation.plot_coordinate_system(self.figure)
        self.canvas.draw()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = My_window()
    window.show()
    sys.exit(app.exec_())
