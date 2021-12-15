import h5py
from scipy import ndimage
import numpy as np
import os

file = (
    "/cluster/work/cvl/esandstroem/data/replica/manual/hotel_0/proxy_alpha_hotel_0.hdf"
)

# open grid
# read from hdf file!
f = h5py.File(file, "r")
voxels = np.array(f["proxy_alpha"]).astype(np.float32)

# apply median filter
voxels = ndimage.median_filter(voxels, size=7, mode="reflect")

# save hdf file
with h5py.File(
    os.path.join("/".join(file.split("/")[:-1]), "proxy_alpha_hotel_0_median7.hdf"), "w"
) as hf:
    hf.create_dataset(
        "proxy_alpha",
        shape=voxels.shape,
        data=voxels,
        compression="gzip",
        compression_opts=9,
    )
