#!/bin/bash

PROJECT_PATH='/home/esandstroem/scratch-second/opportunistic_3d_capture/data/scene3d/stonewall'

# extract features
# colmap feature_extractor --image_path $PROJECT_PATH/images \
#                          --database_path $PROJECT_PATH/database.db \
#                          --ImageReader.camera_model PINHOLE \
#                          --ImageReader.single_camera 1 \

# # # sequential matching along trajectory
# colmap sequential_matcher --database_path $PROJECT_PATH/database.db \
#                           --SequentialMatching.overlap 10

# ## dense reconstruction
# mkdir -p $PROJECT_PATH/dense/sparse

# # build sparse model
# colmap point_triangulator --database_path $PROJECT_PATH/database.db \
#                           --image_path $PROJECT_PATH/images \
#                           --input_path $PROJECT_PATH/sparse \
#                           --output_path $PROJECT_PATH/dense/sparse \
#                           --Mapper.ba_refine_focal_length 0 \
#                           --Mapper.ba_refine_extra_param 0

# # # create dense workspace folders
# cp -r $PROJECT_PATH/images $PROJECT_PATH/dense/
# mkdir -p $PROJECT_PATH/dense/stereo/depth_maps
# mkdir -p $PROJECT_PATH/dense/stereo/normal_maps


# # # compute dense depth maps
# colmap patch_match_stereo --workspace_path $PROJECT_PATH/dense 

# # # fuse stereo depth maps
colmap stereo_fusion --workspace_path PROJECT_PATH/dense \
                     --output_path PROJECT_PATH/dense/fused.ply
