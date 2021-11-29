import matplotlib.pyplot as plt
import numpy as np


def visualize_voxels(voxels: np.ndarray, axes: plt.Axes = None, show=True):
    if axes is None:
        axes: plt.Axes = plt.figure().add_subplot(projection='3d')
    axes.voxels(voxels)

    if show:
        plt.show()


def visualize_points(points: np.ndarray, axes: plt.Axes = None, show=True):
    if axes is None:
        axes: plt.Axes = plt.figure().add_subplot(projection='3d')

    if points.shape[1] == 4:
        c = np.zeros([points.shape[0], 4], dtype=float)
        c[:, 2] = 1.0
        c[:, 3] = points[:, 3]
    else:
        c = None

    axes.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)
    axes.set_xlim3d(left=-1.0, right=1.0)
    axes.set_ylim3d(bottom=-1.0, top=1.0)
    axes.set_zlim3d(bottom=-1.0, top=1.0)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")

    if show:
        plt.show()
