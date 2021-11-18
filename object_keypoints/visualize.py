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
    axes.scatter(points[:, 0], points[:, 1], points[:, 2])
    axes.set_xlim3d(left=-1.0, right=1.0)
    axes.set_ylim3d(bottom=-1.0, top=1.0)
    axes.set_zlim3d(bottom=-1.0, top=1.0)

    if show:
        plt.show()
