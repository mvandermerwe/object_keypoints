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

    if show:
        plt.show()
