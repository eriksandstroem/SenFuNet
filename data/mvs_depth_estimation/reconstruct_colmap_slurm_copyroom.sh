#!/bin/bash

#SBATCH --output=/scratch_net/nudel/colmap-test/log/%j.out  # could not get it to work on nudel_second... #/scratch_ned/nudel_second/opportunistic_3d_capture/data/scene3d/log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G


PROJECT_PATH='/home/esandstroem/scratch-second/opportunistic_3d_capture/data/scene3d'

scene_string='copyroom_downsampled'

for SCENE in $scene_string
do
	# extract features
	colmap feature_extractor --image_path $PROJECT_PATH/$SCENE/images \
	                         --database_path $PROJECT_PATH/$SCENE/database.db \
	                         --ImageReader.camera_model PINHOLE \
	                         --ImageReader.single_camera 1 \
	                         --ImageReader.camera_params "525.0, 525.0, 319.5, 239.5"

	# # sequential matching along trajectory
	colmap sequential_matcher --database_path $PROJECT_PATH/$SCENE/database.db \
	                          --SequentialMatching.overlap 10

	## dense reconstruction
	mkdir -p $PROJECT_PATH/$SCENE/dense/sparse

	# build sparse model
	colmap point_triangulator --database_path $PROJECT_PATH/$SCENE/database.db \
	                          --image_path $PROJECT_PATH/$SCENE/images \
	                          --input_path $PROJECT_PATH/$SCENE/sparse \
	                          --output_path $PROJECT_PATH/$SCENE/dense/sparse \
	                          --Mapper.ba_refine_focal_length 0 \
	                          --Mapper.ba_refine_extra_param 0

	# # create dense workspace folders
	cp -r $PROJECT_PATH/$SCENE/images $PROJECT_PATH/$SCENE/dense/
	mkdir -p $PROJECT_PATH/$SCENE/dense/stereo/depth_maps
	mkdir -p $PROJECT_PATH/$SCENE/dense/stereo/normal_maps


	# # compute dense depth maps
	colmap patch_match_stereo --workspace_path $PROJECT_PATH/$SCENE/dense \
							  --PatchMatchStereo.depth_min 0.5 \
							  --PatchMatchStereo.depth_max 10.0

	# # # fuse stereo depth maps
	# colmap stereo_fusion --workspace_path PROJECT_PATH/${SCENE}/dense \
	#                      --output_path PROJECT_PATH/${SCENE}/dense/fused.ply
done


