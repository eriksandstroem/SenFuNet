import os

scene = 'lounge'

data_path = '/home/esandstroem/scratch-second/euler_work/data/scene3d' + '/' + scene

remove_list = []

image_list = sorted(os.listdir(data_path + '/images'))
tof_list = sorted(os.listdir(data_path + '/' + scene + '_png/depth'))

for k, frame in enumerate(sorted(os.listdir(data_path + '/images'))):
	if k % 10 != 0:
		remove_list.append(data_path + '/images/' + image_list[k])
		remove_list.append(data_path + '/' + scene + '_png/depth/' + tof_list[k])


for path in remove_list:
	os.system('rm ' + path)
