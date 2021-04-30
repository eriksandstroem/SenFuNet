# This script extracts the training, testing or validation data to the compute node on the cluster
# depending on the selected mode. It takes as input the path to the config file and the selected mode.
import os
import sys
path_to_loading_module =  '/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/utils/'
sys.path.append(path_to_loading_module)
import loading

def retrieve_scenes(config_path, mode):
	# retrieve path to scene list
	config = loading.load_config_from_yaml(config_path)
	if mode == 'train':
		scene_path = config.DATA.train_scene_list
	elif mode == 'val':
		scene_path = config.DATA.val_scene_list
	elif mode == 'test':
		scene_path = config.DATA.test_scene_list

	# retrive list of scenes to extract on cluster node
	scenes = []
	with open(scene_path, 'r') as file:
		for line in file:
			line = line.split(' ')
			scene = line[0].split('/')[0]
			if scene not in scenes:
				scenes.append(scene)

	returnstring = ''
	for scene in scenes:
		returnstring += scene
		returnstring += ' '

	print(returnstring) # this is how I return the string
