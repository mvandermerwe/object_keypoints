import matplotlib.pyplot as plt
import numpy as np


def visualize_voxels(voxels: np.ndarray, axes: plt.Axes = None):
    if axes is None:
        axes: plt.Axes = plt.figure().add_subplot(projection='3d')
    axes.voxels(voxels)
    plt.show()
