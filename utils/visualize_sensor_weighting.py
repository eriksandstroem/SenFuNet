import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def visualize_sensor_weighting(
    tsdf,
    sensor_weighting,
    test_dir,
    mask,
    truncation,
    length,
    max_resolution,
    resolution,
    voxel_size,
    outlier_channel,
):
    cmap = plt.get_cmap("inferno")

    if outlier_channel:
        sensor_weighting = sensor_weighting[0, :, :, :]

    hist = sensor_weighting[mask].flatten()

    cm = plt.get_cmap("inferno")
    n, bins, patches = plt.hist(hist, bins=100)
    for c, p in zip(bins, patches):
        plt.setp(p, "facecolor", cm(c))
    plt.savefig(test_dir + "/sensor_weighting_grid_histogram_no_outlier_filter.png")
    plt.clf()

    # Create the mesh using the given mask
    tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
    tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = tsdf

    indices_x = mask.nonzero()[0]
    indices_y = mask.nonzero()[1]
    indices_z = mask.nonzero()[2]

    volume = o3d.integration.UniformTSDFVolume(
        length=length,
        resolution=max_resolution,
        sdf_trunc=truncation,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(indices_x.shape[0]):
        volume.set_tsdf_at(
            tsdf_cube[indices_x[i], indices_y[i], indices_z[i]],
            indices_x[i],
            indices_y[i],
            indices_z[i],
        )
        volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()

    del volume
    mesh.compute_vertex_normals()

    # read vertices from mesh
    vertices = mesh.vertices

    # we need to subtract half a voxel size from the vertices to get to the voxel points
    # since the marching cubes algorithm of open3d thinks that the tsdf voxel vertices are
    # always located at the mid point between the metric space resolution i.e. if we have a tsdf
    # grid of shape 2,2,2 and a voxel size of 1, the marching cubes algorithm will generate a surface at 0.5, 0.5, 0.5
    # to 1.5, 1.5, 1.5.
    voxel_points = np.round(
        np.asarray(vertices) * 1 / voxel_size - voxel_size / 2
    ).astype(int)

    # remove voxels if they are outside of the voxelgrid - these are treated as uninitialized.
    # this step is not needed when we subtract half a voxel size - without this the transformation
    # is wrong.
    valid_points = (
        (voxel_points[:, 0] >= 0)
        * (voxel_points[:, 0] < sensor_weighting.shape[0])
        * (voxel_points[:, 1] >= 0)
        * (voxel_points[:, 1] < sensor_weighting.shape[1])
        * (voxel_points[:, 2] >= 0)
        * (voxel_points[:, 2] < sensor_weighting.shape[2])
    )
    filtered_voxel_points = voxel_points[valid_points, :]

    vals = -np.ones(voxel_points.shape[0])
    vals[valid_points] = sensor_weighting[
        filtered_voxel_points[:, 0],
        filtered_voxel_points[:, 1],
        filtered_voxel_points[:, 2],
    ]
    colors = cmap((vals * 255).astype(int))[:, :-1]

    if (vals == -1).sum() > 0:
        print("Invalid index or indices found among voxel points!")
    # print((vals == -1).sum()) # this sum should always be zero when we subtract half a voxel size to get to the voxel
    # coordinate space.
    colors[vals == -1] = [0, 1, 0]  # make all uninitialized voxels green
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(
        test_dir + "/sensor_weighting_nn_no_outlier_filter.ply", mesh
    )

    # compute surface histogram
    n, bins, patches = plt.hist(vals.flatten(), bins=100)
    for c, p in zip(bins, patches):
        plt.setp(p, "facecolor", cm(c))
    plt.savefig(test_dir + "/sensor_weighting_surface_histogram_no_outlier_filter.png")
    plt.clf()
