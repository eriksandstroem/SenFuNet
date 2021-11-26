import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_sensor_weighting(tsdf, sensor_weighting, test_dir, mask, truncation, length, max_resolution, resolution, voxel_size, outlier_channel):
	cmap = plt.get_cmap("inferno")
	# print weighting histogram - only works in the 2-sensor case! With more sensors we cannot do this in the same way
	# bad idea to not feed the mask to this function as we will include wrongful values in the histogram which are 
	# not in the initialized mask as we update the entire filtered grid chunks during test time. This means that 
	# the filtered grid and the sensor_weighting grid will be filled with ones even though those voxels are uninitialized.
	# this skews our perception of the grid histogram.
	# hist = sensor_weighting[sensor_weighting > -1].flatten()
	if outlier_channel:
		sensor_weighting = sensor_weighting[0, :, :, :]
		
	hist = sensor_weighting[mask].flatten()

	cm = plt.get_cmap('inferno')
	n, bins, patches = plt.hist(hist, bins = 100)
	for c, p in zip(bins, patches):
		plt.setp(p, 'facecolor', cm(c))
	plt.savefig(test_dir + '/sensor_weighting_grid_histogram_no_outlier_filter.png')
	plt.clf()


	# Create the mesh using the given mask
	tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
	tsdf_cube[:resolution[0], :resolution[1], :resolution[2]] = tsdf


	indices_x = mask.nonzero()[0]
	indices_y = mask.nonzero()[1]
	indices_z = mask.nonzero()[2]

	# this creates a voxelgrid with max_resolution voxels along the length length. Each 
	# voxel consists of 8 vertices in the tsdf_cube which means that when we have a tsdf_cube
	# of max_resolution 2 (8 vertices), we will make the uniform volume of size 27 vertices.
	# This is not a problem, however, since we will only initialize the valid indices. I.e. 
	# the unifor volue is always 1 vertex layer too large compared to the tsdf_cube. To correct
	# for this, the max_resolution variable should be 1 less than it is now, making length smaller
	# as well since length is max_resolution times voxel_size
	volume = o3d.integration.UniformTSDFVolume(
			length=length,
			resolution=max_resolution,
			sdf_trunc=truncation,
			color_type=o3d.integration.TSDFVolumeColorType.RGB8)
	
	for i in range(indices_x.shape[0]):
		volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
		volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])               

	print("Extract a triangle mesh from the volume and visualize it.")
	mesh = volume.extract_triangle_mesh()

	del volume
	mesh.compute_vertex_normals()

	# read vertices from mesh
	vertices = mesh.vertices

	# cube = np.ones((2,2,2))
	# index_cube = np.asarray(cube.nonzero()).transpose()

	# # Traverse vertices - not good because it smooths the alpha values too much I think, but I will try it and see.
	# valList = list()
	# for k, vertex in enumerate(vertices):
	# 	# convert vertex from metric space to voxel space
	# 	voxel_point = np.expand_dims(np.floor(vertex * 1/voxel_size).astype(int), 0)
	# 	index_hypothesis = index_cube + voxel_point.repeat(8, 0)
	# 	indices_filtered = []
	# 	for index in index_hypothesis:
	# 		if index[0] < sensor_weighting.shape[0] and index[1] < sensor_weighting.shape[1] and index[2] < sensor_weighting.shape[2]:
	# 			indices_filtered.append(index)

	# 	indices_filtered = np.array(indices_filtered)
	# 	# print(indices_filtered)

	# 	vals = sensor_weighting[indices_filtered[:, 0], indices_filtered[:, 1], indices_filtered[:, 2]]
	# 	_bool = vals > -1 
	# 	# print(conf_bool)
	# 	# print(vals)
	# 	vals = vals[_bool]
	# 	# print(_bool)
	# 	# print(vals)
	# 	val = vals.sum()/_bool.sum()
	# 	# print(val)
	# 	valList.append(val)

		
	# 	# color_conf = cmap(p * val + m) # here I can compute a linear function from the minimum value to the maximum so as to use the full colorband
	# 	# print(val*255)
	# 	color = cmap(int(val*255)) # color range is from 0-255 as integers apparently
	# 	# print(color[:-1])
	# 	# print(colors.shape)
	# 	# print(dir(mesh))
	# 	# mesh.vertex_colors[k] = o3d.utility.Vector2dVector(np.array(color[:-1])*255)
	# 	mesh.vertex_colors[k] = np.array(color[:-1]) #[color[0], color[1], color[2], color[3]]

	# o3d.io.write_triangle_mesh(test_dir + '/sensor_weighting.ply', mesh)

	# print('len vallist ', len(valList))

	# we need to subtract half a voxel size from the vertices to get to the voxel points 
	# since the marching cubes algorithm of open3d thinks that the tsdf voxel vertices are
	# always located at the mid point between the metric space resolution i.e. if we have a tsdf
	# grid of shape 2,2,2 and a voxel size of 1, the marching cubes algorithm will generate a surface at 0.5, 0.5, 0.5
	# to 1.5, 1.5, 1.5.
	voxel_points = np.round(np.asarray(vertices) * 1/voxel_size - voxel_size/2).astype(int)

	# remove voxels if they are outside of the voxelgrid - these are treated as uninitialized. 
	# this step is not needed when we subtract half a voxel size - without this the transformation 
	# is wrong.
	valid_points = (voxel_points[:, 0] >= 0) * (voxel_points[:, 0] < sensor_weighting.shape[0]) * \
		(voxel_points[:, 1] >= 0) * (voxel_points[:, 1] < sensor_weighting.shape[1]) * \
		(voxel_points[:, 2] >= 0) * (voxel_points[:, 2] < sensor_weighting.shape[2])
	filtered_voxel_points = voxel_points[valid_points, :]

	vals = -np.ones(voxel_points.shape[0])
	vals[valid_points] = sensor_weighting[filtered_voxel_points[:, 0], filtered_voxel_points[:, 1], filtered_voxel_points[:, 2]]
	colors = cmap((vals*255).astype(int))[:, :-1]
	# print(colors.shape)
	if (vals == -1).sum() > 0:
		print('Invalid index or indices found among voxel points!')
		# return
	# print((vals == -1).sum()) # this sum should always be zero when we subtract half a voxel size to get to the voxel
	# coordinate space.
	colors[vals == -1] = [0, 1, 0] # make all uninitialized voxels green
	# print(np.asarray(mesh.vertex_colors).shape)
	# print(colors.shape)
	mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
	o3d.io.write_triangle_mesh(test_dir + '/sensor_weighting_nn_no_outlier_filter.ply', mesh)

	# compute surface histogram
	n, bins, patches = plt.hist(vals.flatten(), bins = 100)
	for c, p in zip(bins, patches):
		plt.setp(p, 'facecolor', cm(c))
	plt.savefig(test_dir + '/sensor_weighting_surface_histogram_no_outlier_filter.png')
	plt.clf()

