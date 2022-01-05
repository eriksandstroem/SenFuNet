import os

scene = 'copyroom'

data_path = '/home/esandstroem/scratch-second/opportunistic_3d_capture/data/scene3d' + '/' + scene + '_downsampled'

remove_list = []
image_list = sorted(os.listdir(data_path + '/images'))
cfg = data_path + '/dense/stereo/patch-match_new.cfg'
images = data_path + '/sparse/images_new.txt'
tof_list = sorted(os.listdir(data_path + '/' + scene + '_png/depth'))

with open(data_path + '/dense/stereo/patch-match.cfg', 'r') as cfg_file, \
		open(cfg, 'w') as cfg_file_new, \
		open(data_path + '/sparse/images.txt', 'r') as traj_file, \
		open(images, 'w') as traj_file_new:

	cfg_file = cfg_file.readlines()
	traj_file = traj_file.readlines()

	for k, frame in enumerate(image_list):
		if k % 10 != 0:
			# pass
			remove_list.append(data_path + '/images/' + image_list[k])
			remove_list.append(data_path + '/' + scene + '_png/depth/' + tof_list[k])
		else:
			traj_file_new.write(str(k//10 + 1) + ' ' + ' '.join(traj_file[2*k].split(' ')[1:]))
			traj_file_new.write('\n')
			cfg_file_new.write(cfg_file[2*k])
			cfg_file_new.write(cfg_file[2*k + 1])


for path in remove_list:
	os.system('rm ' + path)

# remove old patch-match.cfg and images.txt
os.system('rm ' + data_path + '/dense/stereo/patch-match.cfg')
os.system('rm ' + data_path + '/sparse/images.txt')

# rename new files to old names
os.system('mv ' + data_path + '/sparse/images_new.txt' + ' ' + data_path + '/sparse/images.txt')
os.system('mv ' + data_path + '/dense/stereo/patch-match_new.cfg' + ' ' + data_path + '/dense/stereo/patch-match.cfg')