import torch
import argparse
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import loading
from utils import setup


from modules.pipeline import Pipeline

from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Script for creating a video of the reconstruction process."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)

    args = parser.parse_args()

    return vars(args)


def test_fusion(config, scene):

    # Note the reason why we don't see any surface at the back in the beginning of the trajectory is because
    # all 8 vertices of a voxel need to have a non-zero weight counter in order for us to have a
    # a surface registered by the marching cubes algorithm. This is why we see surface closer to the camera
    # i.e. because here the rays are denser and they "activate" all corners of the voxel while further away
    # the rays are more sparse and it requires a few frames until all corners are initialized. I know that
    # there is surface registration happening at the far wall, because I evaluated the script with the
    # final weight mask and then we can see that wall.

    option_file = "/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/videos/render_option_old.json"
    # not necessarily used
    transform_file = (
        "/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/videos/transform_"
        + scene
        + ".txt"
    )

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    config.SETTINGS.device = device
    config.FUSION_MODEL.device = device

    # get test dataset
    data_config = setup.get_data_config(config, mode="test")
    dataset = setup.get_data(config.DATA.dataset, data_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.TESTING.test_batch_size,
        shuffle=config.TESTING.test_shuffle,
        pin_memory=True,
        num_workers=16,
    )

    # specify number of features
    if config.FEATURE_MODEL.use_feature_net:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
        )
    else:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.append_depth + 3 * config.FEATURE_MODEL.w_rgb
        )  # 1 for label encoding of noise in gaussian threshold data

    # get test database
    database = setup.get_database(dataset, config, mode="test")

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    loading.load_pipeline(
        config.TESTING.fusion_model_path, pipeline
    )  # this loads all parameters it can

    # create empty PinholeCameraIntrinsic object
    intrinsic_obj = o3d.camera.PinholeCameraIntrinsic()

    # T = np.loadtxt(transform_file)
    origin = np.transpose(np.array([database.scenes_gt[scene].origin]))
    T = np.eye(3)
    T = np.concatenate((T, origin), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)

    pipeline.eval()

    sensors = config.DATA.input

    save_sensor = "fused"  # or sensor name or 'fused', 'weighting'

    # define output dir
    model = config.TESTING.fusion_model_path.split("/")[-3]
    output_folder = "/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/videos/"
    output_folder += model + "/" + scene + "/" + save_sensor

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    voxel_size = 0.01
    truncation = 0.05

    with torch.no_grad():
        for k, batch in tqdm(enumerate(loader), total=len(dataset), mininterval=30):

            # fusion pipeline
            pipeline.test_step_video(batch, database, save_sensor, sensors, device)
            # probably I should not do the outlier filter in an accumulated fashion!

            intrinsic_obj.set_intrinsics(
                512, 512, 512 / 2, 512 / 2, 512 / 2 - 0.5, 512 / 2 - 0.5
            )

            extrinsics = np.array(batch["extrinsics"]).squeeze()
            if extrinsics.shape[0] == 4:
                extrinsics = np.linalg.inv(extrinsics)
            else:
                extrinsics = np.linalg.inv(
                    np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
                )

            # create empty PinholeCameraParameters object
            camera_obj = o3d.camera.PinholeCameraParameters()
            camera_obj.extrinsic = extrinsics
            camera_obj.intrinsic = intrinsic_obj

            # if k == 503:
            #     print(extrinsics)

            # create PinholeCameraTrajectory from the camera list
            trajectory = o3d.camera.PinholeCameraTrajectory()
            trajectory.parameters = [camera_obj]

            # create mesh of database grids
            if save_sensor == "fused":
                mesh = get_mesh_fused(
                    database, T, sensors, voxel_size, scene, truncation
                )
            elif save_sensor == "weighting":
                mesh = get_mesh_weighting(
                    database, T, sensors, voxel_size, scene, truncation
                )
            else:
                mesh = get_mesh(save_sensor, database, T, voxel_size, scene, truncation)

            fixed_camera = o3d.camera.PinholeCameraParameters()
            # office 0 camera
            fixed_camera.extrinsic = [
                [4.69471574e-01, 8.82947564e-01, -8.76720940e-10, 7.06485510e-01],
                [3.01985890e-01, -1.60568759e-01, -9.39692616e-01, 1.60801813e-01],
                [-8.29699337e-01, 4.41158980e-01, -3.42020184e-01, 3.11112356e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
            # human
            # rotation = Quaternion(-0.0732, -0.4448, -0.0178, 0.8925).rotation_matrix
            # translation = np.transpose(np.array([[1.5, 1.2271, -1.8818]]))

            # extrinsics = np.linalg.inv(np.concatenate((np.concatenate((rotation, translation), axis=1), np.array([[0, 0, 0, 1]])), axis=0))
            # fixed_camera.extrinsic = extrinsics

            # copyroom
            # fixed_camera.extrinsic = np.linalg.inv([[-0.2331990000,   -0.6390280000,   0.7329810000,    -origin[0, 0]-3.5],
            #                             [-0.0787258000,   0.7636960000,    0.6407580000,    -origin[1, 0]-1.8],
            #                             [-0.9692370000,   0.0917198000,    -0.2284010000,   -origin[2, 0]-1.0],
            #                             [0.0000000000,    0.0000000000,    0.0000000000,    1.0000000000]])

            fixed_camera.intrinsic = intrinsic_obj

            def custom_draw_geometry_with_camera_trajectory(mesh):
                custom_draw_geometry_with_camera_trajectory.index = -1
                custom_draw_geometry_with_camera_trajectory.trajectory = trajectory
                custom_draw_geometry_with_camera_trajectory.vis = (
                    o3d.visualization.Visualizer()
                )

                def move_forward(vis):
                    # This function is called within the o3d.visualization.Visualizer::run() loop
                    # The run loop calls the function, then re-render
                    # So the sequence in this function is to:
                    # 1. Capture frame
                    # 2. index++, check ending criteria
                    # 3. Set camera
                    # 4. (Re-render)
                    ctr = vis.get_view_control()

                    glb = custom_draw_geometry_with_camera_trajectory

                    if glb.index >= 0:

                        # print("Capture image {:05d}".format(glb.index))
                        image = vis.capture_screen_float_buffer(False)
                        plt.imsave(
                            output_folder + "/" + "%04d" % k + ".png",
                            np.asarray(image),
                            dpi=1,
                        )

                    glb.index = glb.index + 1
                    if glb.index < len(glb.trajectory.parameters):
                        # load fixed camera view
                        ctr.convert_from_pinhole_camera_parameters(fixed_camera)
                    else:
                        custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(
                            None
                        )
                        vis.destroy_window()
                    return False

                vis = custom_draw_geometry_with_camera_trajectory.vis
                vis.create_window(width=512, height=512, visible=False)
                vis.add_geometry(mesh)

                camera = o3d.geometry.LineSet()
                camera = camera.create_camera_visualization(
                    300,
                    300,
                    trajectory.parameters[0].intrinsic.intrinsic_matrix,
                    trajectory.parameters[0].extrinsic,
                    scale=0.25,
                )
                vis.add_geometry(camera)
                vis.get_render_option().load_from_json(option_file)
                vis.register_animation_callback(move_forward)
                vis.run()

            custom_draw_geometry_with_camera_trajectory(mesh)

    # create video of the rendered images
    os.system(
        "ffmpeg -framerate 30 -i "
        + output_folder
        + "/%05d_"
        + save_sensor
        + ".png -vcodec libx264 -preset veryslow -c:a libmp3lame -r 15 "
        + output_folder
        + ".mp4"
    )


def get_mesh(save_sensor, database, transform, voxel_size, scene, truncation):
    # only implemented using Open3D marching cubes
    resolution = database[scene]["tsdf_" + save_sensor].shape
    max_resolution = np.array(resolution).max()
    length = (max_resolution) * voxel_size

    tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
    tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = database[scene][
        "tsdf_" + save_sensor
    ].numpy()

    weights = database[scene]["weights_" + save_sensor].numpy()  # .astype(np.float16)
    mask = weights > 0

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

    # add vertex coloring
    mesh.paint_uniform_color(np.array([0.6, 0.6, 0.6]))

    mesh.transform(transform)
    mesh.compute_vertex_normals()

    return mesh


def get_mesh_weighting(database, transform, sensors, voxel_size, scene, truncation):
    # only implemented using Open3D marching cubes
    resolution = database[scene]["filtered"].shape
    max_resolution = np.array(resolution).max()
    length = (max_resolution) * voxel_size

    tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
    tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = database[scene][
        "filtered"
    ].numpy()

    sensor_weighting_mesh_mask = np.zeros_like(database[scene]["filtered"])
    for sensor_ in sensors:
        sensor_weighting_mesh_mask = np.logical_or(
            sensor_weighting_mesh_mask > 0,
            database[scene]["weights_" + sensor_].numpy() > 0,
        )  # .astype(np.float16)

    indices_x = sensor_weighting_mesh_mask.nonzero()[0]
    indices_y = sensor_weighting_mesh_mask.nonzero()[1]
    indices_z = sensor_weighting_mesh_mask.nonzero()[2]

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

    sensor_weighting_mesh = volume.extract_triangle_mesh()

    # add vertex coloring -
    cmap = plt.get_cmap("inferno")
    voxel_points = np.round(
        np.asarray(sensor_weighting_mesh.vertices - voxel_size / 2) * 1 / voxel_size
    ).astype(int)

    sensor_weighting = database[scene]["sensor_weighting"]
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
    # print(colors.shape)
    if (vals == -1).sum() > 0:
        print("Invalid index or indices found among voxel points!")
        # return
    # print((vals == -1).sum()) # this sum should always be zero when we subtract half a voxel size to get to the voxel
    # coordinate space.
    colors[vals == -1] = [0, 1, 0]  # make all uninitialized voxels green

    sensor_weighting_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    sensor_weighting_mesh.transform(transform)
    sensor_weighting_mesh.compute_vertex_normals()

    return sensor_weighting_mesh


def get_mesh_fused(database, transform, sensors, voxel_size, scene, truncation):
    # only implemented using Open3D marching cubes
    resolution = database[scene]["filtered"].shape
    max_resolution = np.array(resolution).max()
    length = (max_resolution) * voxel_size

    tsdf_cube = torch.zeros((max_resolution, max_resolution, max_resolution))
    tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = database[scene][
        "filtered"
    ]

    mesh_mask = torch.zeros_like(database[scene]["filtered"])
    and_mask = torch.ones_like(database[scene]["filtered"])
    sensor_mask = dict()
    for sensor_ in sensors:
        mesh_mask = torch.logical_or(
            mesh_mask > 0, database[scene]["weights_" + sensor_] > 0
        )  # .astype(np.float16)
        weights = database[scene][
            "weights_" + sensor_
        ]  # database.fusion_weights[sensor_][scene]
        and_mask = torch.logical_and(and_mask, weights > 0)
        sensor_mask[sensor_] = weights > 0

    sensor_weighting = database[scene]["sensor_weighting"]

    only_one_sensor_mask = torch.logical_xor(mesh_mask, and_mask)
    for sensor_ in sensors:
        only_sensor_mask = torch.logical_and(only_one_sensor_mask, sensor_mask[sensor_])
        if sensor_ == sensors[0]:
            rem_indices = torch.logical_and(only_sensor_mask, sensor_weighting < 0.5)
        else:
            rem_indices = torch.logical_and(only_sensor_mask, sensor_weighting > 0.5)

        mesh_mask[rem_indices] = 0

    indices = mesh_mask.nonzero()

    volume = o3d.integration.UniformTSDFVolume(
        length=length,
        resolution=max_resolution,
        sdf_trunc=truncation,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(indices.shape[0]):
        volume.set_tsdf_at(
            tsdf_cube[indices[i, 0], indices[i, 1], indices[i, 2]],
            indices[i, 0],
            indices[i, 1],
            indices[i, 2],
        )
        volume.set_weight_at(1, indices[i, 0], indices[i, 1], indices[i, 2])

    fused_mesh = volume.extract_triangle_mesh()

    # add vertex coloring -
    fused_mesh.paint_uniform_color(np.array([0.6, 0.6, 0.6]))

    fused_mesh.transform(transform)
    fused_mesh.compute_vertex_normals()

    return fused_mesh


if __name__ == "__main__":

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args["config"])

    test_fusion(test_config, args["scene"])
