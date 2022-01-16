import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

from openTSNE import TSNE

import trimesh
import math
import os

import matplotlib

matplotlib.use("Agg")


def visualize_features(
    proxy_sensor_weighting,
    sensor_weighting,
    fused_tsdf,
    gt_tsdf,
    tsdfs,
    weights,
    features,
    test_dir,
    voxel_size,
    truncation,
    scene,
    mc,
):

    # since I am stuck with the cpu accelerated t-SNE version for now, I need to speed it up more, by
    # removing points that we feed to the t-SNE algorithm. I do this by downsampling the grids to get a
    # larger voxel size. With voxel size 1 cm, I have around 12M samples to be embedded. With downsampling
    # factor 2 this should drop.

    downsampling_factor = 2

    # define normalization for the scatter plots so that no normalization takes place
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

    mesh_paths = dict()
    features_both_sensors = None
    error_per_sensor = dict()

    tsne = TSNE(
        perplexity=3,
        learning_rate="auto",
        metric="euclidean",
        n_jobs=-1,  # use all cores
        negative_gradient_method="fft",
        random_state=42,
    )

    # downsample grids
    gt_tsdf = gt_tsdf[
        ::downsampling_factor, ::downsampling_factor, ::downsampling_factor
    ]
    fused_tsdf = fused_tsdf[
        ::downsampling_factor, ::downsampling_factor, ::downsampling_factor
    ]
    sensor_weighting = sensor_weighting[
        ::downsampling_factor, ::downsampling_factor, ::downsampling_factor
    ]
    proxy_sensor_weighting = proxy_sensor_weighting[
        ::downsampling_factor, ::downsampling_factor, ::downsampling_factor
    ]
    break_point = 0
    for k, sensor_ in enumerate(tsdfs.keys()):
        print("Processing: ", sensor_)
        tsdfs[sensor_] = tsdfs[sensor_][
            ::downsampling_factor, ::downsampling_factor, ::downsampling_factor
        ]

        # create mesh
        resolution = tsdfs[sensor_].shape
        max_resolution = np.array(resolution).max()
        v_size = voxel_size * downsampling_factor
        length = (max_resolution) * v_size

        tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
        tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = tsdfs[sensor_]

        weights[sensor_] = weights[sensor_][
            ::downsampling_factor, ::downsampling_factor, ::downsampling_factor
        ]
        mask = weights[sensor_] > 0

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

        mesh = volume.extract_triangle_mesh()

        del volume
        mesh.compute_vertex_normals()
        mesh_paths[sensor_] = (
            test_dir + "/downsampled_" + sensor_ + "_" + scene + ".ply"
        )
        # o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(
            test_dir + "/downsampled_" + sensor_ + "_" + scene + ".ply", mesh
        )  # will remove this written mesh later

        # My reason for using trimesh here and not open3d was that trimesh can save rgba colors while open3d cant. But
        # it turns out that meshlab cannot display rgba colors so this is useless, but I keep it for now.
        mesh = trimesh.load_mesh(
            mesh_paths[sensor_]
        )  # critical to load this, otherwise the colors will be messed up!
        vertices = mesh.vertices

        voxel_points = np.round(np.asarray(vertices) * 1 / v_size - v_size / 2).astype(
            int
        )

        features[sensor_] = features[sensor_][
            ::downsampling_factor, ::downsampling_factor, ::downsampling_factor, :
        ]

        error_per_sensor[sensor_] = np.abs(
            tsdfs[sensor_][voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
            - gt_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
        )
        if k == 0:
            features_both_sensors = features[sensor_][
                voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :
            ]
            break_point = features_both_sensors.shape[0]
        else:
            features_both_sensors = np.concatenate(
                (
                    features_both_sensors,
                    features[sensor_][
                        voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :
                    ],
                ),
                axis=0,
            )

    # compute tsne visualization for the inter sensor study
    print("Compute inter sensor t-SNE")
    print("Input shape: ", features_both_sensors.shape)

    # No conversion to float32 since this can make 0.0 std into non-zero somehow, but it often leads to overflow so I have to do it anyway.
    features_both_sensors = features_both_sensors.astype(np.float32)
    if (
        np.std(features_both_sensors[:, 0]) == 0
        and np.std(features_both_sensors[:, 1]) == 0
    ):
        print("No variance in features for both sensors")
        for sensor_ in features.keys():
            os.system("rm " + mesh_paths[sensor_])  # clean up
        return
    # compute tsne embedding of both sensors to compare inter sensor variation of features
    tsne_result = tsne.fit(features_both_sensors)

    # get colors for the tsne embedding of both sensors
    colors_both_sensors = get_colors_from_tsne_embedding(tsne_result)

    # split the colors to each sensor specific grid
    color_per_sensor = dict()
    tsne_per_sensor = dict()

    # put colors from the inter sensor study in color grids and separete the per sensor tsne embeddings
    for k, sensor_ in enumerate(features.keys()):
        color_per_sensor[sensor_] = dict()
        if k == 0:
            color_per_sensor[sensor_]["norm"] = colors_both_sensors["norm"][
                :break_point
            ]
            color_per_sensor[sensor_]["phase"] = colors_both_sensors["phase"][
                :break_point
            ]
            tsne_per_sensor[sensor_] = tsne_result[:break_point]
        else:
            color_per_sensor[sensor_]["norm"] = colors_both_sensors["norm"][
                break_point:
            ]
            color_per_sensor[sensor_]["phase"] = colors_both_sensors["phase"][
                break_point:
            ]
            tsne_per_sensor[sensor_] = tsne_result[break_point:]

    # plot tsne data with sensor labels

    # colorize each feature with the error of the tdf prediction at that voxel
    sensor_to_color = {list(features.keys())[0]: "lightgreen"}
    if len(list(features.keys())) > 1:
        sensor_to_color[list(features.keys())[1]] = "darkgreen"

    # plot each category with a distinct label
    fig, ax = plt.subplots(1, 1)
    for sensor_, color in sensor_to_color.items():
        max_dist = 0.05  # controls where we threshold
        c = np.array(error_per_sensor[sensor_]) / max_dist
        c[
            c > 0.85
        ] = 0.85  # controls at what color we threshold / strange that I had to adjust this to get yellow. Normally
        # this is 0.85 for yellow. Ahh... this is because of the normalization that takes place I guess!
        c += 0.33  # controls where 0 error is - green
        c[c > 1] = c[c > 1] - 1

        data = tsne_per_sensor[sensor_]
        ax.scatter(data[:, 0], data[:, 1], s=0.1, c=c, cmap="hsv", norm=normalize)

    plt.savefig(
        test_dir + "/" + scene + "_tsne_visualization_both_sensors_color_error.png"
    )
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    for sensor_, color in sensor_to_color.items():
        data = tsne_per_sensor[sensor_]
        # compute the phase of all points
        phase = np.angle(data[:, 0] + 1j * data[:, 1])
        # the output from phase is a number between -pi to pi. This
        # needs to be transformed to 0-1 for the cmap
        phase = phase / math.pi  # range -1 to 1
        phase = (phase + 1) / 2  # range 0-1

        ax.scatter(data[:, 0], data[:, 1], s=0.1, c=phase, cmap="hsv", norm=normalize)

    plt.savefig(
        test_dir + "/" + scene + "_tsne_visualization_both_sensors_color_phase.png"
    )
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    for sensor_, color in sensor_to_color.items():
        data = tsne_per_sensor[sensor_]
        norm = np.linalg.norm(data, axis=1)
        max_norm = np.amax(norm)
        # when I only visualize the features directly without tsne,
        # since I normalize the features individually, this step serves no purpose
        # as the norm for all features equals the max_norm which is one.

        norm = (
            norm / max_norm
        )  # here we have the alpha value for all points. Now they are in range 0-1

        ax.scatter(
            data[:, 0], data[:, 1], s=0.1, c=norm, cmap="inferno", norm=normalize
        )

    plt.savefig(
        test_dir + "/" + scene + "_tsne_visualization_both_sensors_color_norm.png"
    )
    plt.clf()
    plt.close()

    # if the categories overlap, it is tricky to see each one
    # thus plot the categories individually as well
    # plot each category with a distinct label
    for sensor_, color in sensor_to_color.items():
        fig, ax = plt.subplots(1, 1)
        data = tsne_per_sensor[sensor_]
        max_dist = 0.05  # controls where we threshold
        c = np.array(error_per_sensor[sensor_]) / max_dist
        c[
            c > 0.85
        ] = 0.85  # controls at what color we threshold / strange that I had to adjust this to get yellow. Normally
        # this is 0.85 for yellow. Ahh... this is because of the normalization that takes place I guess!
        c += 0.33  # controls where 0 error is - green
        c[c > 1] = c[c > 1] - 1
        ax.scatter(data[:, 0], data[:, 1], s=0.1, c=c, cmap="hsv", norm=normalize)

        plt.savefig(
            test_dir
            + "/"
            + scene
            + "_tsne_visualization_both_sensors_"
            + sensor_
            + "color_error.png"
        )
        plt.clf()
        plt.close()

    for sensor_, color in sensor_to_color.items():
        fig, ax = plt.subplots(1, 1)
        data = tsne_per_sensor[sensor_]
        # compute the phase of all points
        phase = np.angle(data[:, 0] + 1j * data[:, 1])
        # the output from phase is a number between -pi to pi. This
        # needs to be transformed to 0-1 for the cmap
        phase = phase / math.pi  # range -1 to 1
        phase = (phase + 1) / 2  # range 0-1
        ax.scatter(data[:, 0], data[:, 1], s=0.1, c=phase, cmap="hsv")

        plt.savefig(
            test_dir
            + "/"
            + scene
            + "_tsne_visualization_both_sensors_"
            + sensor_
            + "color_phase.png"
        )
        plt.clf()
        plt.close()

    for sensor_, color in sensor_to_color.items():
        fig, ax = plt.subplots(1, 1)
        data = tsne_per_sensor[sensor_]
        norm = np.linalg.norm(data, axis=1)
        max_norm = np.amax(norm)
        # when I only visualize the features directly without tsne,
        # since I normalize the features individually, this step serves no purpose
        # as the norm for all features equals the max_norm which is one.

        norm = (
            norm / max_norm
        )  # here we have the alpha value for all points. Now they are in range 0-1

        ax.scatter(
            data[:, 0], data[:, 1], s=0.1, c=norm, cmap="inferno", norm=normalize
        )

        plt.savefig(
            test_dir
            + "/"
            + scene
            + "_tsne_visualization_both_sensors_"
            + sensor_
            + "color_norm.png"
        )
        plt.clf()
        plt.close()

    # colorize the per sensor meshes with the color that is inter sensor consistent
    for sensor_ in features.keys():
        # load mesh using trimesh since trimesh can save RGBA colors per vertex. Yes, but I cannot display them in meshlab
        # so print to two meshes instead
        mesh_phase = trimesh.load_mesh(mesh_paths[sensor_])
        mesh_norm = trimesh.load_mesh(mesh_paths[sensor_])

        mesh_phase.visual.vertex_colors = color_per_sensor[sensor_]["phase"]
        mesh_phase.remove_degenerate_faces()
        mesh_phase.export(
            test_dir
            + "/"
            + scene
            + "_inter_feature_visualization_"
            + sensor_
            + "_phase.ply"
        )

        mesh_norm.visual.vertex_colors = color_per_sensor[sensor_]["norm"]
        mesh_norm.remove_degenerate_faces()
        mesh_norm.export(
            test_dir
            + "/"
            + scene
            + "_inter_feature_visualization_"
            + sensor_
            + "_norm.ply"
        )
        os.system("rm " + mesh_paths[sensor_])  # clean up

    # get visualization of the input feature space to the weighting network. This is
    # essentially what determines how discriminative it is. The feature visualization per
    # sensor does not really tell us anything.

    # get filtered mesh. This is the union mesh without any heuristic outlier filtering so that
    # we can observe the features at these locations

    # get union mask
    mask = np.zeros_like(list(weights.keys())[0], dtype=np.bool_)
    and_mask = np.ones_like(list(weights.keys())[0], dtype=np.bool_)
    sensor_mask = dict()
    for sensor_ in features.keys():
        sensor_mask[sensor_] = weights[sensor_] > 0
        mask = np.logical_or(
            mask, weights[sensor_] > 0
        )  # this is the union without heuristic outlier filter
        and_mask = np.logical_and(
            and_mask, weights[sensor_] > 0
        )  # this is the union without heuristic outlier filter

    # create mesh
    resolution = weights[list(weights.keys())[0]].shape
    max_resolution = np.array(resolution).max()
    v_size = voxel_size * downsampling_factor
    length = (max_resolution) * v_size

    indices_x = mask.nonzero()[0]
    indices_y = mask.nonzero()[1]
    indices_z = mask.nonzero()[2]

    volume = o3d.integration.UniformTSDFVolume(
        length=length,
        resolution=max_resolution,
        sdf_trunc=truncation,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8,
    )

    # not necessary to make the tsdf_cube, but I did it anyways now
    tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
    tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = fused_tsdf

    for i in range(indices_x.shape[0]):
        volume.set_tsdf_at(
            tsdf_cube[indices_x[i], indices_y[i], indices_z[i]],
            indices_x[i],
            indices_y[i],
            indices_z[i],
        )
        volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])

    mesh = volume.extract_triangle_mesh()

    # get vertices
    vertices = mesh.vertices

    voxel_points = np.round(np.asarray(vertices) * 1 / v_size - v_size / 2).astype(int)

    # get features at these voxel points
    features_both_sensors = None
    for sensor_ in features.keys():
        if features_both_sensors is None:
            features_both_sensors = features[sensor_][
                voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :
            ]
        else:
            features_both_sensors = np.concatenate(
                (
                    features_both_sensors,
                    features[sensor_][
                        voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :
                    ],
                ),
                axis=-1,
            )

    # compute tsne embedding
    tsne_result = tsne.fit(features_both_sensors)
    # get colors for the tsne embedding of both sensors
    colors = get_colors_from_tsne_embedding(tsne_result)

    # plot the tsne embedding with the per voxel error, phase, norm, alpha and proxy alpha
    fig, ax = plt.subplots(1, 1)
    error = np.abs(
        fused_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
        - gt_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
    )
    max_dist = 0.05  # controls where we threshold
    c = np.array(error) / max_dist
    c[
        c > 0.85
    ] = 0.85  # controls at what color we threshold / strange that I had to adjust this to get yellow. Normally
    # this is 0.85 for yellow
    c += 0.33  # controls where 0 error is - green
    c[c > 1] = c[c > 1] - 1

    ax.scatter(
        tsne_result[:, 0], tsne_result[:, 1], s=0.1, c=c, cmap="hsv", norm=normalize
    )  # , label=sensor_)

    # this plots the error of the fused result, but this is also not that interesting. We want to plot
    # the features with the relative error of one of the sensors so that we can see if there is a
    # correlation between the sensor error and the alpha. I also need to plot the alpha as coloring!
    plt.savefig(test_dir + "/" + scene + "_tsne_visualization_fused_color_error.png")
    plt.clf()
    plt.close()

    # plot the error visualization again, but now the points which belong to predicted outliers. We
    # do this to see how many of the yellow outliers we catch with the heuristic outlier filter and where they are
    # in the 2D plane
    fig, ax = plt.subplots(1, 1)
    error = np.abs(
        fused_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
        - gt_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
    )
    max_dist = 0.05  # controls where we threshold
    c = np.array(error) / max_dist
    c[c > 0.85] = 0.85
    c += 0.33  # controls where 0 error is - green
    c[c > 1] = c[c > 1] - 1

    only_one_sensor_mask = np.logical_xor(mask, and_mask)

    for sensor_ in features.keys():
        only_sensor_mask = np.logical_and(only_one_sensor_mask, sensor_mask[sensor_])
        if sensor_ == list(features.keys())[0]:
            rem_indices = np.logical_and(only_sensor_mask, sensor_weighting < 0.5)

        else:
            # before I fixed the bug always ended up here when I had tof and stereo as sensors
            # but this would mean that for the tof sensor I removed those indices
            # if alpha was larger than 0.5 which it almost always is. This means that
            # essentially all (cannot be 100 % sure) voxels where we only integrated
            # tof, was removed. Since the histogram is essentially does not have
            # any voxels with trust less than 0.5, we also removed all alone stereo voxels
            # so at the end we end up with a mask very similar to the and_mask
            rem_indices = np.logical_and(only_sensor_mask, sensor_weighting > 0.5)

        mask[rem_indices] = 0
    # retrieve the boolean value per voxel point
    mask = mask[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]

    ax.scatter(
        tsne_result[mask, 0],
        tsne_result[mask, 1],
        s=0.1,
        c=c[mask],
        cmap="hsv",
        norm=normalize,
    )

    # this plots the error of the fused result, but this is also not that interesting. We want to plot
    # the features with the relative error of one of the sensors so that we can see if there is a
    # correlation between the sensor error and the alpha. I also need to plot the alpha as coloring!
    plt.savefig(
        test_dir
        + "/"
        + scene
        + "_tsne_visualization_fused_color_error_remove_outliers.png"
    )
    plt.clf()

    fig, ax = plt.subplots(1, 1)
    alpha = sensor_weighting[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
    ax.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        s=0.1,
        c=alpha,
        cmap="inferno",
        norm=normalize,
    )

    plt.savefig(test_dir + "/" + scene + "_tsne_visualization_fused_color_alpha.png")
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    # compute the phase of all points
    phase = np.angle(tsne_result[:, 0] + 1j * tsne_result[:, 1])
    # the output from phase is a number between -pi to pi. This
    # needs to be transformed to 0-1 for the cmap
    phase = phase / math.pi  # range -1 to 1
    phase = (phase + 1) / 2  # range 0-1

    ax.scatter(
        tsne_result[:, 0], tsne_result[:, 1], s=0.1, c=phase, cmap="hsv", norm=normalize
    )  # , label=sensor_)

    plt.savefig(test_dir + "/" + scene + "_tsne_visualization_fused_color_phase.png")
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    norm = np.linalg.norm(tsne_result, axis=1)
    max_norm = np.amax(norm)
    # when I only visualize the features directly without tsne,
    # since I normalize the features individually, this step serves no purpose
    # as the norm for all features equals the max_norm which is one.

    norm = (
        norm / max_norm
    )  # here we have the alpha value for all points. Now they are in range 0-1

    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], s=0.1, c=norm, cmap="inferno")

    plt.savefig(test_dir + "/" + scene + "_tsne_visualization_fused_color_norm.png")
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    alpha = sensor_weighting[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
    ax.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        s=0.1,
        c=alpha,
        cmap="inferno",
        norm=normalize,
    )

    plt.savefig(test_dir + "/" + scene + "_tsne_visualization_fused_color_alpha.png")
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    alpha = proxy_sensor_weighting[
        voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]
    ]
    ax.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        s=0.1,
        c=alpha,
        cmap="inferno",
        norm=normalize,
    )

    plt.savefig(
        test_dir + "/" + scene + "_tsne_visualization_fused_color_proxy_alpha.png"
    )
    plt.clf()
    plt.close()

    # paste the phase and norm colors on the fused mesh
    mesh.compute_vertex_normals()

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors["phase"])
    o3d.io.write_triangle_mesh(
        test_dir + "/" + scene + "_fused_feature_visualization_phase.ply", mesh
    )  # will remove this written mesh later

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors["norm"])

    o3d.io.write_triangle_mesh(
        test_dir + "/" + scene + "_fused_feature_visualization_norm.ply", mesh
    )  # will remove this written mesh later

    # sanity check - put the error as a color on the downsampled mesh and compare to the not downsampled mesh
    error = np.abs(
        fused_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
        - gt_tsdf[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
    )
    max_dist = 0.05  # controls where we threshold
    c = np.array(error) / max_dist
    c[c > 0.85] = 0.85
    c += 0.33  # controls where 0 error is - green
    c[c > 1] = c[c > 1] - 1
    cm = plt.get_cmap("hsv")
    mesh.vertex_colors = o3d.utility.Vector3dVector(cm(c)[:, :-1])

    o3d.io.write_triangle_mesh(
        test_dir + "/" + scene + "_fused_feature_visualization_error.ply", mesh
    )  # will remove this written mesh later


def get_colors_from_tsne_embedding(tsne_result):
    cmap_phase = plt.get_cmap(
        "hsv"
    )  # it is important to use a cyclic color map since we want the color map to be
    cmap_norm = plt.get_cmap(
        "inferno"
    )  # this is to display the norm of the tsne result
    # continuous when the vector loops around

    # find the maximum length vector and normalize with this value
    # compute norm over all data points
    norm = np.linalg.norm(tsne_result, axis=1)
    max_norm = np.amax(norm)
    # when I only visualize the features directly without tsne,
    # since I normalize the features individually, this step serves no purpose
    # as the norm for all features equals the max_norm which is one.

    norm = (
        norm / max_norm
    )  # here we have the alpha value for all points. Now they are in range 0-1
    norm_colors = cmap_norm(norm)

    # compute the phase of all points
    phase = np.angle(tsne_result[:, 0] + 1j * tsne_result[:, 1])
    # the output from phase is a number between -pi to pi. This
    # needs to be transformed to 0-1 for the cmap
    phase = phase / math.pi  # range -1 to 1
    phase = (phase + 1) / 2  # range 0-1

    # get the color from the colormap
    phase_colors = cmap_phase(phase)

    colors = dict()
    colors["norm"] = norm_colors[:, :3]
    colors["phase"] = phase_colors[:, :3]
    return colors
