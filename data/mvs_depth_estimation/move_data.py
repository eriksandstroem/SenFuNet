import os

scene = 'cactusgarden'

data_target_path = '/home/esandstroem/scratch-second/euler_work/data/scene3d' + '/' + scene
data_source_path = '/home/esandstroem/scratch-second/opportunistic_3d_capture/data/scene3d' + '/' + scene

copy_list = dict()

image_list = sorted(os.listdir(data_source_path + '/images'))
stereo_list = sorted(os.listdir(data_source_path + '/dense/stereo/depth_maps'))

# remove all entries containing 'photometric' from the stereo_list
stereo_list = stereo_list[::2]

tof_list = sorted(os.listdir(data_source_path + '/' + scene + '_png/depth'))

for k, frame in enumerate(sorted(os.listdir(data_source_path + '/images'))):
	if k % 10 == 0:
		copy_list[data_source_path + '/images/' + image_list[k]] = data_target_path + '/images/' + image_list[k]
		copy_list[data_source_path + '/dense/stereo/depth_maps/' + stereo_list[k]] = data_target_path + '/dense/stereo/depth_maps/' + stereo_list[k]
		copy_list[data_source_path + '/' + scene + '_png/depth/' + tof_list[k]] = data_target_path + '/' + scene + '_png/depth/' + tof_list[k]

for path in copy_list.keys():
	os.system('cp ' + path + ' ' + copy_list[path])