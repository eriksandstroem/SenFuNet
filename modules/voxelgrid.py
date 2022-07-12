import numpy as np
import math


class FeatureGrid(object):
    def __init__(self, voxel_size, n_features, bbox=None):

        self._resolution = voxel_size
        self._bbox = bbox
        self._n_features = n_features
        self._volume = None

        if bbox is not None:
            self._origin = bbox[:, 0]

            volume_shape = np.diff(self._bbox, axis=1).ravel() / self.resolution
            # float16 conversion critical - otherwise, numerical
            # instabilies will cause wrong voxel grid size
            volume_shape = volume_shape.astype(np.float16)
            self._shape = (
                np.ceil([volume_shape[0], volume_shape[1], volume_shape[2], n_features])
                .astype(np.int32)
                .tolist()
            )  # round up

            self._volume = np.zeros(self._shape, dtype=np.float16)

    @property
    def resolution(self):
        return self._resolution

    @property
    def bbox(self):
        assert self._bbox is not None
        return self._bbox

    @property
    def volume(self):
        assert self._volume is not None
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = volume

    @property
    def origin(self):
        assert self._origin is not None
        return self._origin

    @property
    def shape(self):
        assert self._volume is not None
        return self._volume.shape

    def __getattr__(self, x, y, z):
        return self._volume[x, y, z, :]


class VoxelGrid(object):
    def __init__(self, voxel_size, volume=None, bbox=None, initial_value=0.0):

        self._resolution = voxel_size

        self._volume = volume
        self._bbox = bbox

        if bbox is not None:
            self._origin = bbox[:, 0]

        if volume is None and bbox is not None:
            volume_shape = np.diff(self._bbox, axis=1).ravel() / self.resolution
            # float16 conversion critical - otherwise, numerical
            # instabilies will cause wrong voxel grid size
            volume_shape = volume_shape.astype(np.float16)

            volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()  # round up
            # float 16 conversion is critical
            self._volume = initial_value * np.ones(volume_shape).astype("float16")

    def from_array(self, array, bbox):

        self._volume = array
        self._bbox = bbox
        self._origin = bbox[:, 0]

    @property
    def resolution(self):
        return self._resolution

    @property
    def bbox(self):
        assert self._bbox is not None
        return self._bbox

    @property
    def volume(self):
        assert self._volume is not None
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = volume

    @property
    def origin(self):
        assert self._origin is not None
        return self._origin

    @property
    def shape(self):
        assert self._volume is not None
        return self._volume.shape

    def __getattr__(self, x, y, z):
        return self._volume[x, y, z]
