import numpy as np
import transforms3d as tf3d


def point_cloud_to_voxel(point_cloud, voxel_size):
    """
    Point cloud (in range [-1, 1]) to voxel.

    Args:
        point_cloud: point cloud
        voxel_size: number of voxels along each dim
    """
    voxel_length = 2.0 / voxel_size
    voxel_locs = (point_cloud / voxel_length).astype(int) + (voxel_size // 2)

    voxels = np.zeros([voxel_size, voxel_size, voxel_size], dtype=bool)
    voxel_locs = np.clip(voxel_locs, a_min=0, a_max=voxel_size - 1)
    voxels[(voxel_locs[:, 0], voxel_locs[:, 1], voxel_locs[:, 2])] = True

    return voxels


def rotate_point_cloud(point_cloud, euler):
    """
    Rotate provided point cloud by provided euler angles.

    Args:
        point_cloud: point cloud
        euler: euler angle rotation to apply.
    """
    rot_matrix = tf3d.euler.euler2mat(euler[0], euler[1], euler[2])

    rotated_points = (rot_matrix @ point_cloud.T).T
    return rotated_points, rot_matrix


def random_rotate_point_cloud(point_cloud):
    """
    Apply a random rotation to the provided point cloud.

    Args:
        point_cloud: point cloud
    """
    random_euler = -np.pi + (2.0 * np.pi * np.random.random(3))
    return rotate_point_cloud(point_cloud, random_euler)


def random_z_rotate_point_cloud(point_cloud):
    """
    Apply a random rotation around z to the provided point cloud.

    Args:
        point_cloud: point cloud
    """
    rand_z_rot = -np.pi + (2.0 * np.pi * np.random.random())
    euler = [0.0, 0.0, rand_z_rot]
    return rotate_point_cloud(point_cloud, euler)


def scale_point_cloud(point_cloud, scale):
    """
    Apply the provided anistropic scaling to the point cloud.

    Args:
        point_cloud: point cloud
        scale: anistropic scaling terms along each dim
    """
    return scale * point_cloud, scale


def random_scale_point_cloud(point_cloud, min_scale=0.75):
    """
    Apply random anistropic scaling between [min_scale, 1.0]
    along each dim.

    Args:
        point_cloud: point cloud
        min_scale: minimum scaling to allow
    """
    scale = min_scale + (np.random.random(3) * (1 - min_scale))
    return scale_point_cloud(point_cloud, scale)
