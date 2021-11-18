import numpy as np
import transforms3d as tf3d


class PointCloudToVoxel(object):
    """
    Point cloud (in range [-1, 1]) to voxel.
    """

    def __init__(self, point_cloud_key: str, voxel_size: int, voxel_key: str):
        self.point_cloud_key = point_cloud_key
        self.voxel_size = voxel_size
        self.voxel_key = voxel_key

    def __call__(self, data):
        new_data = {}
        for key in data.keys():
            new_data[key] = data[key]

        pc = data[self.point_cloud_key]
        voxel_length = 2.0 / self.voxel_size
        voxel_locs = (pc / voxel_length).astype(int) + (self.voxel_size // 2)

        new_voxel = np.zeros([self.voxel_size, self.voxel_size, self.voxel_size], dtype=bool)
        for voxel_loc in voxel_locs:
            new_voxel[np.clip(voxel_loc[0], 0, self.voxel_size-1),
                      np.clip(voxel_loc[1], 0, self.voxel_size-1),
                      np.clip(voxel_loc[2], 0, self.voxel_size-1)] = True
        new_data[self.voxel_key] = new_voxel

        return new_data


class TransformPointCloud(object):

    def __init__(self, point_cloud_key: str):
        self.point_cloud_key = point_cloud_key

    def __call__(self, data):
        new_data = {}
        for key in data.keys():
            new_data[key] = data[key]

        # Build random transform.
        random_euler = -np.pi + (2.0 * np.pi * np.random.random(3))
        rot = tf3d.euler.euler2mat(random_euler[0], random_euler[1], random_euler[2])

        # Apply to point cloud.
        pc = data[self.point_cloud_key]
        new_pc = rot @ pc.T

        new_data['rot'] = random_euler
        new_data['tf_point_cloud'] = new_pc.T
        return new_data


class ScalePointCloud(object):

    def __init__(self, point_cloud_key: str):
        self.point_cloud_key = point_cloud_key

    def __call__(self, data):
        new_data = {}
        for key in data.keys():
            new_data[key] = data[key]

        # Get random scale.
        scale = 0.75 + (np.random.random(3) * 0.25)

        # Apply to point cloud.
        pc = data[self.point_cloud_key]
        new_pc = scale * pc

        new_data['scale'] = scale
        new_data['tf_point_cloud'] = new_pc
        return new_data
