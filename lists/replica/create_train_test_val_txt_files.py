import os

## REPLICA
root_left_RGB = '/home/esandstroem/scratch-second/opportunistic_3d_capture/data/habitat/replica/manual'
root_left_GT = '/home/esandstroem/scratch-second/opportunistic_3d_capture/data/habitat/replica/manual'
filename_train = 'train.txt'
filename_test = 'test.txt'
filename_val = 'val.txt'

# I just put some scenes here for now
depth_scenes_train = ['apartment_1', 'frl_apartment_0', 'office_1', 'room_2', 'office_3', 'room_0']
depth_scenes_test = ['apartment_2', 'office_0', 'hotel_0', 'frl_apartment_2', 'office_4']
depth_scenes_val = ['frl_apartment_1']

#format
# office_0_non_watertight/6/left_depth_gt office_0_non_watertight/6/left_depth_noise_5.0 
#office_0_non_watertight/6/left_bts_depth_finetune_office0 office_0_non_watertight/6/left_psmnet_depth_finetune_office0_bsize3_epoch15 
#office_0_non_watertight/6/left_oracle_fused_depth office_0_non_watertight/6/left_rgb office_0_non_watertight/6/left_camera_matrix

# create new files when I have the mono and stereo depths! Below I just put left_depth_gt for mono and stereo depth.

# TRAIN
file = open(filename_train,'w')
for scene in depth_scenes_train:
	for trajectory in os.listdir(root_left_RGB + '/' + scene):
		if not trajectory.endswith('.hdf'):
		 
			file.write(scene + '/' + trajectory + '/left_depth_gt')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_depth_noise_5.0')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_bts_depth') # MONO DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_psmnet_depth') # STEREO DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_oracle_depth') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_rgb') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_camera_matrix') # ORACLE DEPTH
			file.write('\n')

file.close() 

# TEST
file = open(filename_test,'w') 
for scene in depth_scenes_test:
	for trajectory in os.listdir(root_left_RGB + '/' + scene):
		if not trajectory.endswith('.hdf'):

			file.write(scene + '/' + trajectory + '/left_depth_gt')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_depth_noise_5.0')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_bts_depth') # MONO DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_psmnet_depth') # STEREO DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_oracle_depth') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_rgb') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_camera_matrix') # ORACLE DEPTH
			file.write('\n')

file.close() 

# VAL
file = open(filename_val,'w') 
for scene in depth_scenes_val:
	for trajectory in os.listdir(root_left_RGB + '/' + scene):
		if not trajectory.endswith('.hdf'):

			file.write(scene + '/' + trajectory + '/left_depth_gt')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_depth_noise_5.0')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_bts_depth') # MONO DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_psmnet_depth') # STEREO DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_oracle_depth') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_rgb') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_camera_matrix') # ORACLE DEPTH
			file.write('\n')
			
file.close() 