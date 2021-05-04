import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_sensor_weighting(mesh, sensor_weighting, test_dir, voxel_size):
	cmap = plt.get_cmap("inferno")
	# print weighting histogram - only works in the 2-sensor case! With more sensors we cannot do this in the same way
	hist = sensor_weighting[sensor_weighting > -1].flatten()
	# hist = sensor_weighting[mask].flatten()
	plt.hist(hist, bins = 100)
	plt.savefig(test_dir + '/sensor_weighting_histogram.png')
	plt.clf()

	# read vertices from mesh
	vertices = mesh.vertices

	cube = np.ones((2,2,2))
	index_cube = np.asarray(cube.nonzero()).transpose()

	# Traverse vertices - not good because it smooths the alpha values too much I think, but I will try it and see.
	valList = list()
	for k, vertex in enumerate(vertices):
		# convert vertex from metric space to voxel space
		voxel_point = np.expand_dims(np.floor(vertex * 1/voxel_size).astype(int), 0)
		index_hypothesis = index_cube + voxel_point.repeat(8, 0)
		indices_filtered = []
		for index in index_hypothesis:
			if index[0] < sensor_weighting.shape[0] and index[1] < sensor_weighting.shape[1] and index[2] < sensor_weighting.shape[2]:
				indices_filtered.append(index)

		indices_filtered = np.array(indices_filtered)
		# print(indices_filtered)

		vals = sensor_weighting[indices_filtered[:, 0], indices_filtered[:, 1], indices_filtered[:, 2]]
		_bool = vals > -1 
		# print(conf_bool)
		# print(vals)
		vals = vals[_bool]
		# print(_bool)
		# print(vals)
		val = vals.sum()/_bool.sum()
		# print(val)
		valList.append(val)

		
		# color_conf = cmap(p * val + m) # here I can compute a linear function from the minimum value to the maximum so as to use the full colorband
		# print(val*255)
		color = cmap(int(val*255)) # color range is from 0-255 as integers apparently
		# print(color[:-1])
		# print(colors.shape)
		# print(dir(mesh))
		# mesh.vertex_colors[k] = o3d.utility.Vector2dVector(np.array(color[:-1])*255)
		mesh.vertex_colors[k] = np.array(color[:-1]) #[color[0], color[1], color[2], color[3]]

	o3d.io.write_triangle_mesh(test_dir + '/sensor_weighting.ply', mesh)

	# hist = sensor_weighting[mask].flatten()
	plt.hist(np.array(valList), bins = 100)
	plt.savefig(test_dir + '/sensor_weighting_vertex_histogram.png') # here I see that the averaged alpha values in the neighborhood skews the
	# distribution of the alpha values a lot. I will try with taking the value of the nearest neighbor as well to compare. Then I don't shift
	# the alpha distribution at all
	plt.clf()

	voxel_points = np.round(np.asarray(vertices) * 1/voxel_size).astype(int)
	# print(voxel_points.shape)
	vals = sensor_weighting[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]]
	colors = cmap((vals*255).astype(int))[:, :-1]
	# print(colors.shape)
	colors[vals == -1] = [0, 1, 0]
	# print(np.asarray(mesh.vertex_colors).shape)
	# print(colors.shape)
	mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
	o3d.io.write_triangle_mesh(test_dir + '/sensor_weighting_nn.ply', mesh)

