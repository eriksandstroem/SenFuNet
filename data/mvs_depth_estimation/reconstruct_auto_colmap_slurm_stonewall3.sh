#!/bin/bash

#SBATCH --output=/scratch_net/nudel/colmap-test/log/%j.out  # could not get it to work on nudel_second... #/scratch_ned/nudel_second/opportunistic_3d_capture/data/scene3d/log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G


PROJECT_PATH='/home/esandstroem/scratch-second/opportunistic_3d_capture/data/scene3d'

scene_string='stonewall3'

# The project folder must contain a folder "images" with all the images.
DATASET_PATH='/home/esandstroem/scratch-second/opportunistic_3d_capture/data/scene3d/stonewall3'

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images
   --ImageReader.camera_model PINHOLE \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_params "525.0, 525.0, 319.5, 239.5"

colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_param 0

mkdir $DATASET_PATH/dense

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

$ colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply