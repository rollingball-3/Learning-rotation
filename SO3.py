import numpy as np
from scipy.linalg import logm, expm


class SO3:
    def __init__(self, rotation_matrix=None):
        """
        初始化SO3类

        参数：
        rotation_matrix: 旋转矩阵
        """
        if not isinstance(rotation_matrix, np.ndarray):
            self.rotation_matrix = np.eye(3)
        else:
            if isinstance(rotation_matrix, np.ndarray) and rotation_matrix.shape == (3, 3):
                self.rotation_matrix = rotation_matrix
            else:
                raise ValueError("Invalid rotation matrix. Must be a numpy 3x3 matrix.")

    def __str__(self):
        """
        打印SO3对象的信息
        """
        output = "Rotation Matrix:\n" + "{}\n".format(np.round(self.rotation_matrix, decimals=4))

        quaternion = self.to_quaternion()
        output += "Quaternion:\n" + "{}\n".format(np.round(quaternion, decimals=4))
        theta = 2 * np.arccos(quaternion[0])
        output += "2 * arccos({}) = {}\n".format(quaternion[0], 2 * np.arccos(quaternion[0]))
        output += "{} / sin(angle / 2) = {}\n".format(quaternion[1:4], quaternion[1:4] / np.sin(theta / 2))

        axis_angle = self.to_axis_angle()
        output += "Axis-Angle:\n{:.4f} {:.4f} {:.4f} {:.4f}\n".format(axis_angle[0], axis_angle[1], axis_angle[2],
                                                                      axis_angle[3])
        output += "Axis-Angle:\n{:.4f} {:.4f} {:.4f}\n".format(axis_angle[0] * axis_angle[3],
                                                               axis_angle[1] * axis_angle[3],
                                                               axis_angle[2] * axis_angle[3])

        skew_matrix = self.rotation_matrix_logarithm()
        output += "skew-symmetric matrix / log(rotation_matrix):\n" + "{}".format(np.round(skew_matrix, decimals=4))

        return output

    def __add__(self, other):
        """
        重载加法运算符（+）

        参数：
        other: 另一个SO3对象

        返回：
        新的SO3对象，表示两个旋转的加法结果
        """
        if isinstance(other, SO3):
            new_rotation_matrix = np.dot(self.rotation_matrix, other.rotation_matrix)
            return SO3(new_rotation_matrix)
        else:
            raise ValueError("Invalid operand. Must be an SO3 object.")

    def __sub__(self, other):
        """
        重载减法运算符（-）

        参数：
        other: 另一个SO3对象

        返回：
        新的SO3对象，表示两个旋转的减法结果
        """
        if isinstance(other, SO3):
            inverse_rotation_matrix = np.linalg.inv(other.rotation_matrix)
            new_rotation_matrix = np.dot(self.rotation_matrix, inverse_rotation_matrix)
            return SO3(new_rotation_matrix)
        else:
            raise ValueError("Invalid operand. Must be an SO3 object.")

    def rotation_matrix_logarithm(self):
        """
        将旋转矩阵的指数形式转换为旋转矩阵

        参数：
        exp_matrix: 旋转矩阵的指数形式

        返回值：
        matrix: 旋转矩阵
        """
        matrix = logm(self.rotation_matrix)

        return matrix.real

    def plot_coordinate_system(self, fig):
        """
        绘制当前SO3实例的坐标系

        """
        ax = fig.add_subplot(111, projection='3d')

        # 绘制世界坐标系
        world_axes_length = 1.0
        ax.quiver(0, 0, 0, world_axes_length, 0, 0, color='r', linestyle='dashed', label='World X-axis')
        ax.quiver(0, 0, 0, 0, world_axes_length, 0, color='g', linestyle='dashed', label='World Y-axis')
        ax.quiver(0, 0, 0, 0, 0, world_axes_length, color='b', linestyle='dashed', label='World Z-axis')

        # 坐标轴单位向量
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # 坐标轴变换到当前SO3的旋转坐标系
        x_axis_transformed = np.dot(self.rotation_matrix, x_axis)
        y_axis_transformed = np.dot(self.rotation_matrix, y_axis)
        z_axis_transformed = np.dot(self.rotation_matrix, z_axis)

        # 绘制旋转坐标系
        rotation_axes_length = 1.5
        ax.quiver(0, 0, 0, x_axis_transformed[0], x_axis_transformed[1], x_axis_transformed[2],
                  color='r', label='Rotated X-axis')
        ax.quiver(0, 0, 0, y_axis_transformed[0], y_axis_transformed[1], y_axis_transformed[2],
                  color='g', label='Rotated Y-axis')
        ax.quiver(0, 0, 0, z_axis_transformed[0], z_axis_transformed[1], z_axis_transformed[2],
                  color='b', label='Rotated Z-axis')
        # 设置坐标轴范围
        ax.set_xlim([-rotation_axes_length, rotation_axes_length])
        ax.set_ylim([-rotation_axes_length, rotation_axes_length])
        ax.set_zlim([-rotation_axes_length, rotation_axes_length])

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 添加图例
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def to_quaternion(self):
        """
        将旋转矩阵转换为四元数

        返回值：
        quaternion: 四元数 [w, x, y, z]
        """
        trace = np.trace(self.rotation_matrix)

        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
            w = 0.25 * S
            x = (self.rotation_matrix[2, 1] - self.rotation_matrix[1, 2]) / S
            y = (self.rotation_matrix[0, 2] - self.rotation_matrix[2, 0]) / S
            z = (self.rotation_matrix[1, 0] - self.rotation_matrix[0, 1]) / S
        elif self.rotation_matrix[0, 0] > self.rotation_matrix[1, 1] and self.rotation_matrix[0, 0] > \
                self.rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + self.rotation_matrix[0, 0] - self.rotation_matrix[1, 1] - self.rotation_matrix[
                2, 2]) * 2  # S = 4 * x
            w = (self.rotation_matrix[2, 1] - self.rotation_matrix[1, 2]) / S
            x = 0.25 * S
            y = (self.rotation_matrix[0, 1] + self.rotation_matrix[1, 0]) / S
            z = (self.rotation_matrix[0, 2] + self.rotation_matrix[2, 0]) / S
        elif self.rotation_matrix[1, 1] > self.rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + self.rotation_matrix[1, 1] - self.rotation_matrix[0, 0] - self.rotation_matrix[
                2, 2]) * 2  # S = 4 * y
            w = (self.rotation_matrix[0, 2] - self.rotation_matrix[2, 0]) / S
            x = (self.rotation_matrix[0, 1] + self.rotation_matrix[1, 0]) / S
            y = 0.25 * S
            z = (self.rotation_matrix[1, 2] + self.rotation_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + self.rotation_matrix[2, 2] - self.rotation_matrix[0, 0] - self.rotation_matrix[
                1, 1]) * 2  # S = 4 * z
            w = (self.rotation_matrix[1, 0] - self.rotation_matrix[0, 1]) / S
            x = (self.rotation_matrix[0, 2] + self.rotation_matrix[2, 0]) / S
            y = (self.rotation_matrix[1, 2] + self.rotation_matrix[2, 1]) / S
            z = 0.25 * S

        quaternion = np.array([w, x, y, z])
        quaternion /= np.linalg.norm(quaternion)

        return quaternion

    def to_axis_angle(self):
        """
        将旋转矩阵转换为轴角表示法

        返回值：
        axis_angle: 包含旋转轴和角度的数组 [x, y, z, angle]
        """

        quat = self.to_quaternion()
        angle = 2 * np.arccos(quat[0])
        if angle < 1e-10:
            axis = np.array([1, 0, 0])
            angle = 0
        else:
            axis = quat[1:4] / np.sin(angle / 2)

        axis_angle = np.zeros(4)
        axis_angle[:3] = axis
        axis_angle[3] = angle

        return axis_angle

    @staticmethod
    def from_quaternion(quaternion):
        """
        从四元数创建SO3对象

        参数：
        quaternion: 四元数 [w, x, y, z]

        返回值：
        SO3对象
        """
        quaternion = quaternion.astype(np.float32)

        quaternion /= np.linalg.norm(quaternion)

        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

        rotation_matrix = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                                    [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
                                    [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])

        return SO3(rotation_matrix)

    @staticmethod
    def from_axis_angle(axis_angle):
        """
        从轴角创建SO3对象

        参数：
        axis_angle: 包含旋转轴和角度的数组 [x, y, z, angle]

        返回值：
        SO3对象
        """
        if len(axis_angle) == 4:

            vector, angle = axis_angle[:3], axis_angle[3]
            vector /= np.linalg.norm(vector)
        elif len(axis_angle) == 3:
            angle = np.linalg.norm(axis_angle)
            vector = axis_angle / angle
        else:
            raise ValueError("Dimension of axis angle should be 3 or 4")

        x, y, z = vector[0], vector[1], vector[2]
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        rotation_matrix = np.array([[t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c]])

        return SO3(rotation_matrix)
