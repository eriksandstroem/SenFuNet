import torch
import argparse
import os

import numpy as np

from utils.loading import load_pipeline, load_config_from_yaml

from modules.pipeline import Pipeline

from utils import setup

from utils.metrics import evaluation

from utils.visualize_sensor_weighting import visualize_sensor_weighting
from utils.visualize_features import visualize_features
from utils.compute_proxy_sensor_weighting import compute_proxy_sensor_weighting_and_mesh

import h5py
import open3d as o3d

from evaluate_3d_reconstruction import run_evaluation


def arg_parse():
    parser = argparse.ArgumentParser(description="Script for testing SenFuNet")

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    return vars(args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_fusion(config):
    # define output dir
    test_path = "/test_no_carving_debug"
    test_dir = (
        config.SETTINGS.experiment_path
        + "/"
        + config.TESTING.fusion_model_path.split("/")[-3]
        + test_path
    )

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
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

    # specify number of features to be stored in feature grid at each voxel location
    if config.FEATURE_MODEL.learned_features:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
        )
    else:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.append_pixel_conf
            + config.FEATURE_MODEL.append_depth
            + 3 * config.FEATURE_MODEL.w_rgb
        )  # 1 for label encoding of noise in gaussian threshold data

    # get test database
    database = setup.get_database(dataset, config, mode="test")

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    # count learnable parameters
    for sensor in config.DATA.input:
        if config.FUSION_MODEL.use_fusion_net:
            print(
                "Fusion Net ",
                sensor,
                ": ",
                count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]),
            )
        print(
            "Feature Net ",
            sensor,
            ": ",
            count_parameters(pipeline.fuse_pipeline._feature_network[sensor]),
        )

    if pipeline.filter_pipeline is not None:
        print(
            "Filtering Net: ",
            count_parameters(pipeline.filter_pipeline._filtering_network),
        )

    # load network parameters from trained model
    if config.FILTERING_MODEL.model != "tsdf_middle_fusion":
        load_pipeline(config.TESTING.fusion_model_path, pipeline)

    # put pipelien in evaluation mode
    pipeline.eval()

    sensors = config.DATA.input

    # test model
    if config.FILTERING_MODEL.model != "tsdf_middle_fusion":
        pipeline.test(loader, dataset, database, sensors, device)
    else:
        pipeline.test_tsdf(loader, dataset, database, sensors, device)

    # save hdf-files of test scenes
    for scene_id in database.scenes_gt.keys():
        database.save(path=test_dir, scene_id=scene_id)

    # compute f-scores and voxelgrid scores for the test scenes and render visualizations
    if config.FILTERING_MODEL.model == "routedfusion":
        evaluate_routedfusion(database, config, test_path)
    else:
        evaluate(database, config, test_path)


def evaluate(database, config, test_path):

    # when testing on data located at local scratch of gpu node
    sdf_gt_path = os.getenv(config.DATA.root_dir)
    # os.getenv returns none when the input does not exist. When
    # it returns none, we want to train on the work folder

    if not sdf_gt_path:
        sdf_gt_path = config.DATA.root_dir
    test_dir = config.TESTING.fusion_model_path.split("/")[:-2]
    test_dir = "/".join(test_dir) + test_path

    # define weight counter thresholds on which we evaluate
    weight_thresholds = config.TESTING.weight_thresholds

    # evaluate each test scene
    for scene in database.scenes_gt.keys():
        tsdf_path = test_dir

        # load ground truth signed distance grid
        sdf_gt = sdf_gt_path + "/" + scene + "/sdf_" + scene + ".hdf"
        f = h5py.File(sdf_gt, "r")
        sdf_gt = np.array(f["sdf"]).astype(np.float16)
        # truncate grid
        truncation = config.DATA.trunc_value
        sdf_gt[sdf_gt >= truncation] = truncation
        sdf_gt[sdf_gt <= -truncation] = -truncation

        # pad gt grid if necessary
        pad = config.DATA.pad
        if pad > 0:
            sdf_gt = np.pad(sdf_gt, pad, "constant", constant_values=-truncation)

        # define voxel side length and resolution
        voxel_size = f.attrs["voxel_size"]
        resolution = sdf_gt.shape

        # largest resolution along any dimesnion
        max_resolution = np.array(resolution).max()
        # largest dimension in meters
        length = (max_resolution) * voxel_size

        # evaluate each weight counter threshold
        for weight_threshold in weight_thresholds:
            model_test = scene + "_weight_threshold_" + str(weight_threshold)
            model_test = model_test + "_filtered"

            # define logger to print voxel grid scores
            logger = setup.get_logger(test_dir, name=model_test)

            # read predicted fused tsdf and weight grids
            tsdf = tsdf_path + "/" + scene + ".tsdf_filtered.hf5"
            f = h5py.File(tsdf, "r")
            tsdf = np.array(f["TSDF_filtered"]).astype(np.float16)

            # declare masks used for outlier filter
            mask = np.zeros_like(tsdf)
            and_mask = np.ones_like(tsdf)
            sensor_mask = dict()

            # compute masks used for outlier filter
            for sensor_ in config.DATA.input:
                weights = tsdf_path + "/" + scene + "_" + sensor_ + ".weights.hf5"
                f = h5py.File(weights, "r")
                weights = np.array(f["weights"]).astype(np.float16)
                mask = np.logical_or(mask, weights > 0)
                and_mask = np.logical_and(and_mask, weights > 0)
                sensor_mask[sensor_] = weights > 0

            if config.TESTING.use_outlier_filter:
                # copy the original mask before outlier filtering since we want to visualize
                # the unfiltered mesh
                sensor_weighting_mask = mask.copy()

                # apply outlier filter
                sensor_weighting = tsdf_path + "/" + scene + ".sensor_weighting.hf5"
                f = h5py.File(sensor_weighting, "r")
                sensor_weighting = np.array(f["sensor_weighting"]).astype(np.float16)

                if config.FILTERING_MODEL.outlier_channel:
                    sensor_weighting = sensor_weighting[1, :, :, :]

                only_one_sensor_mask = np.logical_xor(mask, and_mask)
                for sensor_ in config.DATA.input:
                    only_sensor_mask = np.logical_and(
                        only_one_sensor_mask, sensor_mask[sensor_]
                    )
                    if sensor_ == config.DATA.input[0]:
                        rem_indices = np.logical_and(
                            only_sensor_mask, sensor_weighting < 0.5
                        )
                    else:
                        rem_indices = np.logical_and(
                            only_sensor_mask, sensor_weighting > 0.5
                        )

                    mask[rem_indices] = 0

            # apply masking of voxels if weight_treshold > 0
            weight_mask = np.zeros_like(tsdf)
            for sensor_ in config.DATA.input:
                weights = tsdf_path + "/" + scene + "_" + sensor_ + ".weights.hf5"
                f = h5py.File(weights, "r")
                weights = np.array(f["weights"]).astype(np.float16)
                weight_mask = np.logical_or(weight_mask, weights > weight_threshold)

            # filter away outliers using the weight mask when weight_threshold > 0
            mask = np.logical_and(mask, weight_mask)

            # get voxel grid scores
            eval_results_scene = evaluation(tsdf, sdf_gt, mask)

            # log voxel grid scores
            logger.info("Test Scores for scene: " + scene)
            for key in eval_results_scene:
                logger.info(key + ": " + str(eval_results_scene[key]))

            # Create the mesh using the given mask
            tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
            tsdf_cube[: resolution[0], : resolution[1], : resolution[2]] = tsdf

            # OPEN3D MARCHING CUBES
            # ---------------------------------------------
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
            # o3d.visualization.draw_geometries([mesh])
            o3d.io.write_triangle_mesh(
                os.path.join(test_dir, model_test + ".ply"), mesh
            )
            # ---------------------------------------------

            # Skimage marching cubes
            # ---------------------------------------------
            # import skimage.measure
            # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(tsdf, level=0, spacing=(0.01, 0.01, 0.01), mask=mask)
            # import trimesh
            # mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)

            # mesh.export(os.path.join(test_dir, model_test + '.ply'))
            # ---------------------------------------------

            # TEST CODE FOR MARCHING CUBES INNER WORKINGS. LEAVE FOR NOW
            # # volume = o3d.integration.UniformTSDFVolume(
            # #         length=3.0,
            # #         resolution=3,
            # #         sdf_trunc=1.0,
            # #         color_type=o3d.integration.TSDFVolumeColorType.RGB8)

            # # volume.set_tsdf_at(0.5, 1,1,1)
            # # volume.set_tsdf_at(0.5, 2,1,1)
            # # volume.set_tsdf_at(0.5, 2,1,2)
            # # volume.set_tsdf_at(0.5, 1,1,2)
            # # volume.set_weight_at(1, 1,1,1)
            # # volume.set_weight_at(1, 2,1,1)
            # # volume.set_weight_at(1, 2,1,2)
            # # volume.set_weight_at(1, 1,1,2)
            # # volume.set_weight_at(1, 1,2,1)
            # # volume.set_weight_at(1, 1,2,2)
            # # volume.set_weight_at(1, 2,2,2)
            # # volume.set_weight_at(1, 2,2,1)
            # # volume.set_tsdf_at(-0.5, 1,2,1)
            # # volume.set_tsdf_at(-0.5, 1,2,2)
            # # volume.set_tsdf_at(-0.5, 2,2,2)
            # # volume.set_tsdf_at(-0.5, 2,2,1)

            # # volume.set_weight_at(1, 1,1,0)
            # # volume.set_weight_at(1, 2,1,0)
            # # volume.set_weight_at(1, 2,1,1)
            # # volume.set_weight_at(1, 1,1,1)
            # # volume.set_weight_at(1, 1,2,0)
            # # volume.set_weight_at(1, 1,2,1)
            # # volume.set_weight_at(1, 2,2,1)
            # # volume.set_weight_at(1, 2,2,0)
            # # volume.set_tsdf_at(0.5, 1,1,0)
            # # volume.set_tsdf_at(0.5, 2,1,0)
            # # volume.set_tsdf_at(0.5, 2,1,1)
            # # volume.set_tsdf_at(0.5, 1,1,1)
            # # volume.set_tsdf_at(-0.5, 1,2,0)
            # # volume.set_tsdf_at(-0.5, 1,2,1)
            # # volume.set_tsdf_at(-0.5, 2,2,1)
            # # volume.set_tsdf_at(-0.5, 2,2,0)

            # # a = volume.extract_voxel_grid()
            # # print(a)

            # # mesh = volume.extract_triangle_mesh()
            # # print(np.asarray(mesh.vertices))
            # # print(np.asarray(mesh.faces))
            # # o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + 'test4.ply'), mesh)
            # # volume.set_weight_at(1, 0,1,1)
            # # mesh = volume.extract_triangle_mesh()
            # # o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + 'test2.ply'), mesh)
            # # return

            # Compute the F-score, precision and recall
            ply_path = model_test + ".ply"

            # evaluate F-score
            run_evaluation(ply_path, test_dir, scene)

            # move the logs and plys to the evaluation dir created by the run_evaluation script
            os.system(
                "mv "
                + test_dir
                + "/"
                + model_test
                + ".logs "
                + test_dir
                + "/"
                + model_test
                + "/"
                + model_test
                + ".logs"
            )
            os.system(
                "mv "
                + test_dir
                + "/"
                + model_test
                + ".ply "
                + test_dir
                + "/"
                + model_test
                + "/"
                + model_test
                + ".ply"
            )

            if config.TESTING.visualize_sensor_weighting:
                # Generate visualization of the sensor weighting
                # load weighting sensor grid
                sensor_weighting = tsdf_path + "/" + scene + ".sensor_weighting.hf5"
                f = h5py.File(sensor_weighting, "r")
                sensor_weighting = np.array(f["sensor_weighting"]).astype(np.float16)

                # compute sensor weighting histogram and mesh visualization
                visualize_sensor_weighting(
                    tsdf,
                    sensor_weighting,
                    test_dir,
                    sensor_weighting_mask,
                    truncation,
                    length,
                    max_resolution,
                    resolution,
                    voxel_size,
                    config.FILTERING_MODEL.outlier_channel,
                )

                os.system(
                    "mv "
                    + test_dir
                    + "/sensor_weighting_nn_no_outlier_filter.ply "
                    + test_dir
                    + "/"
                    + model_test
                    + "/sensor_weighting_nn.ply"
                )
                os.system(
                    "mv "
                    + test_dir
                    + "/sensor_weighting_grid_histogram_no_outlier_filter.png "
                    + test_dir
                    + "/"
                    + model_test
                    + "/sensor_weighting_grid_histogram.png"
                )
                os.system(
                    "mv "
                    + test_dir
                    + "/sensor_weighting_surface_histogram_no_outlier_filter.png "
                    + test_dir
                    + "/"
                    + model_test
                    + "/sensor_weighting_surface_histogram.png"
                )

            # evaluate single sensor reconstructions
            if config.TESTING.eval_single_sensors:
                # evaluate each sensor
                for sensor_ in config.DATA.input:
                    model_test = scene + "_weight_threshold_" + str(weight_threshold)
                    model_test = model_test + "_" + sensor_
                    logger = setup.get_logger(test_dir, name=model_test)

                    tsdf = tsdf_path + "/" + scene + "_" + sensor_ + ".tsdf.hf5"
                    weights = tsdf_path + "/" + scene + "_" + sensor_ + ".weights.hf5"

                    # read weight grid
                    f = h5py.File(weights, "r")
                    weights = np.array(f["weights"]).astype(np.float16)

                    # read tsdfs grid
                    f = h5py.File(tsdf, "r")
                    tsdf = np.array(f["TSDF"]).astype(np.float16)

                    # filter according to weight threshold
                    mask = weights > weight_threshold

                    # evaluate voxel grid scores
                    eval_results_scene = evaluation(tsdf, sdf_gt, mask)

                    # log voxel grid scores
                    logger.info("Test Scores for scene: " + scene)
                    for key in eval_results_scene:
                        logger.info(key + ": " + str(eval_results_scene[key]))

                    # Create the mesh using the given mask
                    tsdf_cube = np.zeros(
                        (max_resolution, max_resolution, max_resolution)
                    )
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
                        volume.set_weight_at(
                            1, indices_x[i], indices_y[i], indices_z[i]
                        )

                    print("Extract a triangle mesh from the volume and visualize it.")
                    mesh = volume.extract_triangle_mesh()

                    del volume
                    mesh.compute_vertex_normals()
                    # o3d.visualization.draw_geometries([mesh])
                    o3d.io.write_triangle_mesh(
                        os.path.join(test_dir, model_test + ".ply"), mesh
                    )

                    # Compute the F-score, precision and recall
                    ply_path = model_test + ".ply"

                    # evaluate F-score
                    run_evaluation(ply_path, test_dir, scene)

                    # move the logs and plys to the evaluation dirs
                    os.system(
                        "mv "
                        + test_dir
                        + "/"
                        + model_test
                        + ".logs "
                        + test_dir
                        + "/"
                        + model_test
                        + "/"
                        + model_test
                        + ".logs"
                    )
                    os.system(
                        "mv "
                        + test_dir
                        + "/"
                        + model_test
                        + ".ply "
                        + test_dir
                        + "/"
                        + model_test
                        + "/"
                        + model_test
                        + ".ply"
                    )

                    # evalute the refined tsdf grid if available
                    if config.FILTERING_MODEL.use_refinement:
                        model_test = (
                            scene + "_weight_threshold_" + str(weight_threshold)
                        )
                        model_test = model_test + "_refined_" + sensor_
                        logger = setup.get_logger(test_dir, name=model_test)

                        tsdf = (
                            tsdf_path
                            + "/"
                            + scene
                            + "_"
                            + sensor_
                            + ".tsdf_refined.hf5"
                        )
                        weights = (
                            tsdf_path + "/" + scene + "_" + sensor_ + ".weights.hf5"
                        )

                        # read tsdfs and weight grids
                        f = h5py.File(tsdf, "r")
                        tsdf = np.array(f["TSDF"]).astype(np.float16)

                        f = h5py.File(weights, "r")
                        weights = np.array(f["weights"]).astype(np.float16)

                        # filter mask according to weight threshold
                        mask = weights > weight_threshold

                        # evaluate voxel grid scores
                        eval_results_scene = evaluation(tsdf, sdf_gt, mask)

                        # log voxel grid scores
                        logger.info("Test Scores for scene: " + scene)
                        for key in eval_results_scene:
                            logger.info(key + ": " + str(eval_results_scene[key]))

                        # Create the mesh using the given mask
                        tsdf_cube = np.zeros(
                            (max_resolution, max_resolution, max_resolution)
                        )
                        tsdf_cube[
                            : resolution[0], : resolution[1], : resolution[2]
                        ] = tsdf

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
                            volume.set_weight_at(
                                1, indices_x[i], indices_y[i], indices_z[i]
                            )

                        print(
                            "Extract a triangle mesh from the volume and visualize it."
                        )
                        mesh = volume.extract_triangle_mesh()

                        del volume
                        mesh.compute_vertex_normals()
                        # o3d.visualization.draw_geometries([mesh])
                        o3d.io.write_triangle_mesh(
                            os.path.join(test_dir, model_test + ".ply"), mesh
                        )

                        # # Compute the F-score, precision and recall
                        ply_path = model_test + ".ply"

                        # evaluate F-score
                        run_evaluation(ply_path, test_dir, scene)

                        # # move the logs and plys to the evaluation dirs
                        os.system(
                            "mv "
                            + test_dir
                            + "/"
                            + model_test
                            + ".logs "
                            + test_dir
                            + "/"
                            + model_test
                            + "/"
                            + model_test
                            + ".logs"
                        )
                        os.system(
                            "mv "
                            + test_dir
                            + "/"
                            + model_test
                            + ".ply "
                            + test_dir
                            + "/"
                            + model_test
                            + "/"
                            + model_test
                            + ".ply"
                        )

                if config.TESTING.visualize_features_and_proxy:
                    features = dict()
                    tsdfs = dict()
                    weights = dict()
                    for sensor_ in config.DATA.input:
                        featurename = (
                            tsdf_path + "/" + scene + "_" + sensor_ + ".features.hf5"
                        )
                        f = h5py.File(featurename, "r")
                        features[sensor_] = np.array(f["features"]).astype(np.float16)
                        tsdfname = tsdf_path + "/" + scene + "_" + sensor_ + ".tsdf.hf5"
                        f = h5py.File(tsdfname, "r")
                        tsdfs[sensor_] = np.array(f["TSDF"]).astype(np.float16)
                        weightname = (
                            tsdf_path + "/" + scene + "_" + sensor_ + ".weights.hf5"
                        )
                        f = h5py.File(weightname, "r")
                        weights[sensor_] = np.array(f["weights"]).astype(np.float16)

                    proxy_sensor_weighting = compute_proxy_sensor_weighting_and_mesh(
                        tsdfs, sdf_gt, test_dir, weights, voxel_size, truncation, scene
                    )

                    # load fused tsdf
                    fused_tsdf = tsdf_path + "/" + scene + ".tsdf_filtered.hf5"
                    # read tsdf
                    f = h5py.File(fused_tsdf, "r")
                    fused_tsdf = np.array(f["TSDF_filtered"]).astype(np.float16)

                    sensor_weighting = tsdf_path + "/" + scene + ".sensor_weighting.hf5"
                    f = h5py.File(sensor_weighting, "r")
                    sensor_weighting = np.array(f["sensor_weighting"]).astype(
                        np.float16
                    )

                    if config.FILTERING_MODEL.outlier_channel:
                        sensor_weighting = sensor_weighting[0, :, :, :]

                    visualize_features(
                        proxy_sensor_weighting,
                        sensor_weighting,
                        fused_tsdf,
                        sdf_gt,
                        tsdfs,
                        weights,
                        features,
                        test_dir,
                        voxel_size,
                        truncation,
                        scene,
                    )


def evaluate_routedfusion(database, config, test_path):

    # when testing on data located at local scratch of gpu node
    sdf_gt_path = os.getenv(config.DATA.root_dir)
    # os.getenv returns none when the input does not exist. When
    # it returns none, we want to train on the work folder

    if not sdf_gt_path:
        sdf_gt_path = config.DATA.root_dir
    test_dir = config.TESTING.fusion_model_path.split("/")[:-2]
    test_dir = "/".join(test_dir) + test_path

    # define weight counter thresholds on which we evaluate
    weight_thresholds = config.TESTING.weight_thresholds

    # evaluate each test scene
    for scene in database.scenes_gt.keys():
        tsdf_path = test_dir

        # load ground truth signed distance grid
        sdf_gt = sdf_gt_path + "/" + scene + "/sdf_" + scene + ".hdf"
        f = h5py.File(sdf_gt, "r")
        sdf_gt = np.array(f["sdf"]).astype(np.float16)
        # truncate grid
        truncation = config.DATA.trunc_value
        sdf_gt[sdf_gt >= truncation] = truncation
        sdf_gt[sdf_gt <= -truncation] = -truncation

        # pad gt grid if necessary
        pad = config.DATA.pad
        if pad > 0:
            sdf_gt = np.pad(sdf_gt, pad, "constant", constant_values=-truncation)

        # define voxel side length and resolution
        voxel_size = f.attrs["voxel_size"]
        resolution = sdf_gt.shape

        # largest resolution along any dimesnion
        max_resolution = np.array(resolution).max()
        # largest dimension in meters
        length = (max_resolution) * voxel_size

        # evaluate each weight counter threshold
        for weight_threshold in weight_thresholds:
            # evaluate the model
            sensor_ = config.DATA.input[0]
            model_test = scene + "_weight_threshold_" + str(weight_threshold)
            model_test = model_test + "_" + sensor_
            logger = setup.get_logger(test_dir, name=model_test)

            tsdf = tsdf_path + "/" + scene + "_" + sensor_ + ".tsdf.hf5"
            weights = tsdf_path + "/" + scene + "_" + sensor_ + ".weights.hf5"

            # read weight grid
            f = h5py.File(weights, "r")
            weights = np.array(f["weights"]).astype(np.float16)

            # read tsdfs grid
            f = h5py.File(tsdf, "r")
            tsdf = np.array(f["TSDF"]).astype(np.float16)

            if config.TESTING.routedfusion_nn:
                weights = np.zeros_like(weights)
                for sensor_ in config.DATA.input:
                    # to eval routedfusion on nn mask
                    # we specify the path to the corresponding tsdf fusion model
                    # where the nearest neighbor weight hdf grids are stored
                    weights_path = (
                        config.SETTINGS.experiment_path
                        + "/"
                        + config.TESTING.routedfusion_nn_model
                        + test_path
                        + "/"
                        + scene
                        + "_"
                        + sensor_
                        + ".weights.hf5"
                    )
                    f = h5py.File(weights_path, "r")
                    weights_sensor = np.array(f["weights"]).astype(np.float16)
                    weights = np.logical_or(weights, weights_sensor)

            # filter according to weight threshold
            mask = weights > weight_threshold

            # evaluate voxel grid scores
            eval_results_scene = evaluation(tsdf, sdf_gt, mask)

            # log voxel grid scores
            logger.info("Test Scores for scene: " + scene)
            for key in eval_results_scene:
                logger.info(key + ": " + str(eval_results_scene[key]))

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
            # o3d.visualization.draw_geometries([mesh])
            o3d.io.write_triangle_mesh(
                os.path.join(test_dir, model_test + ".ply"), mesh
            )

            # Compute the F-score, precision and recall
            ply_path = model_test + ".ply"

            # evaluate F-score
            run_evaluation(ply_path, test_dir, scene)

            # move the logs and plys to the evaluation dirs
            os.system(
                "mv "
                + test_dir
                + "/"
                + model_test
                + ".logs "
                + test_dir
                + "/"
                + model_test
                + "/"
                + model_test
                + ".logs"
            )
            os.system(
                "mv "
                + test_dir
                + "/"
                + model_test
                + ".ply "
                + test_dir
                + "/"
                + model_test
                + "/"
                + model_test
                + ".ply"
            )


if __name__ == "__main__":

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = load_config_from_yaml(args["config"])

    # test model
    test_fusion(test_config)
