import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from openTSNE import TSNE
# from openTSNE.callbacks import ErrorLogger
import trimesh
import math
import os

def visualize_features(tsdfs, weights, features, test_dir, voxel_size, truncation):
    # how should I go from 2 features to rgb colors. The most obvious is: get the maximum norm of the features
    # in both grids and normalize with this value. This means later that the vector length is the alpha in 
    # (RGBA). The phase is then the input to the cmap.

    # since I am stuck with the cpu accelerated t-SNE version for now, I need to speed it up more, by
    # removing points that we feed to the t-SNE algorithm. I do this by downsampling the grids to get a
    # larger voxel size. With voxel size 1 cm, I have around 12M samples to be embedded. With downsampling
    # factor 2 this should drop.

    downsampling_factor = 1

    mesh_paths = dict()
    features_both_sensors = None

    tsne = TSNE(
        perplexity=3,
        learning_rate='auto',
        metric="euclidean",
        # callbacks=ErrorLogger(),
        n_jobs=-1, # use all cores
        negative_gradient_method='fft',
        random_state=42,
    )

    # downsample grids
    break_point = 0
    for k, sensor_ in enumerate(tsdfs.keys()):
        tsdfs[sensor_] = tsdfs[sensor_][::downsampling_factor, ::downsampling_factor, ::downsampling_factor]

        # create mesh 
        resolution = tsdfs[sensor_].shape
        max_resolution = np.array(resolution).max()
        v_size = voxel_size*downsampling_factor
        length = (max_resolution)*v_size


        tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
        tsdf_cube[:resolution[0], :resolution[1], :resolution[2]] = tsdfs[sensor_]

        mask = weights[sensor_][::downsampling_factor, ::downsampling_factor, ::downsampling_factor] > 0
        indices_x = mask.nonzero()[0]
        indices_y = mask.nonzero()[1]
        indices_z = mask.nonzero()[2]


        volume = o3d.integration.UniformTSDFVolume(
                length=length,
                resolution=max_resolution,
                sdf_trunc=truncation,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
                        
        for i in range(indices_x.shape[0]):
            volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
            volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])               

        mesh = volume.extract_triangle_mesh()

        del volume
        mesh.compute_vertex_normals()
        mesh_paths[sensor_] = test_dir + '/downsampled_' + sensor_ + '.ply'
        # o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(test_dir + '/downsampled_' + sensor_ + '.ply', mesh) # will remove this written mesh later
   
        mesh = trimesh.load_mesh(mesh_paths[sensor_]) # critical to load this, otherwise the colors will be messed up!
        vertices = mesh.vertices

        voxel_points = np.round(np.asarray(vertices) * 1/v_size - v_size/2).astype(int)
        
        features[sensor_] = features[sensor_][::downsampling_factor, ::downsampling_factor, ::downsampling_factor, :]

        if k == 0:
            features_both_sensors = features[sensor_][voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :]
            break_point = features_both_sensors.shape[0]
        else:
            features_both_sensors = np.concatenate((features_both_sensors, 
                        features[sensor_][voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :]), axis=0)


        # why do I make the input float32? If the variance is below 10 to the power of -7, a 
        # float 16 number will make it zero, but perhaps that is what I actually want. So I removed it now.
        # it appears that the tsne.fit function converts the numbers to float32 anyway...
        X = features[sensor_][voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :] #.astype(np.float32)
        print('Feature input shape: ', X.shape)
        if np.std(X[:, 0]) == 0 and np.std(X[:, 1]) == 0: # I should continue for all dimensions ideally
            np.savetxt(test_dir + '/feature_embedding_' + sensor_ + '_float16.txt', X)
            print('No variance in features for float16 sensor; ', sensor_)
 
        X = X.astype(np.float32)
        if np.std(X[:, 0]) == 0 and np.std(X[:, 1]) == 0: # I should continue for all dimensions ideally
            np.savetxt(test_dir + '/feature_embedding_' + sensor_ + '_float32.txt', X)
            print('No variance in features for float32 sensor; ', sensor_)
            continue
    
        # tsne_result = tsne.fit(X)
        tsne_result = X

        np.savetxt(test_dir + '/feature_embedding_' + sensor_ + '.txt', X)
        np.savetxt(test_dir + '/tsne_embedding_' + sensor_ + '.txt', tsne_result)

        fig, ax = plt.subplots(1,1)
        ax.plot(tsne_result[:, 0], tsne_result[:, 1], 'o', 
                color='lightgreen', label=sensor_)
        ax.legend(loc='best')
        plt.savefig(test_dir + '/tsne_visualization_' + sensor_ + '.png')
        plt.clf()

        # colorize the per sensor meshes with the color that is intra sensor consistent

        # load mesh using trimesh since trimesh can save RGBA colors per vertex
        mesh = trimesh.load_mesh(mesh_paths[sensor_])

        mesh.visual.vertex_colors = (get_colors_from_tsne_embedding(tsne_result) *255).astype(np.uint8)

        mesh.remove_degenerate_faces()

        mesh.export(test_dir + '/intra_feature_visualization_' + sensor_ + '.ply')


    # compute tsne visualization for the inter sensor study
    print('Compute inter sensor t-SNE')
    print('Input shape: ', features_both_sensors.shape)
    
    # No conversion to float32 since this can make 0.0 std into non-zero somehow
    features_both_sensors = features_both_sensors #.astype(np.float32)
    if np.std(features_both_sensors[:, 0]) ==  0 and np.std(features_both_sensors[:, 1]) == 0:
        print('No variance in features for both sensors')
        for sensor_ in features.keys():
            os.system('rm ' + mesh_paths[sensor_]) # clean up
        return
    # compute tsne embedding of both sensors to compare inter sensor variation of features
    # tsne_result = tsne.fit(features_both_sensors)
    tsne_result = features_both_sensors
    # # save tsne data for further analysis
    # np.savetxt(test_dir + '/tsne_embedding.txt', tsne_result)
    # tsne_result = np.loadtxt(test_dir + '/tsne_embedding.txt')
    # get colors for the tsne embedding of both sensors
    colors_both_sensors = get_colors_from_tsne_embedding(tsne_result)
    np.savetxt(test_dir + '/color_both_sensors.txt', colors_both_sensors)

    # split the colors to each sensor specific grid
    color_per_sensor = dict()
    tsne_per_sensor = dict()

    # put colors from the inter sensor study in color grids and separete the per sensor tsne embeddings
    for k, sensor_ in enumerate(features.keys()):
        if k == 0:
            color_per_sensor[sensor_] = colors_both_sensors[:break_point]
            tsne_per_sensor[sensor_] = tsne_result[:break_point]
        else:
            color_per_sensor[sensor_] = colors_both_sensors[break_point:]
            tsne_per_sensor[sensor_] = tsne_result[break_point:]


    # plot tsne data with sensor labels
    # make a mapping from category to your favourite colors and labels
    sensor_to_color = {list(features.keys())[0]: 'lightgreen'}
    if len(list(features.keys())) > 1:
        sensor_to_color[list(features.keys())[1]] = 'darkgreen' 

    # plot each category with a distinct label
    fig, ax = plt.subplots(1,1)
    for sensor_, color in sensor_to_color.items():
        data = tsne_per_sensor[sensor_]
        ax.plot(data[:, 0], data[:, 1], 'o', 
                color=color, label=sensor_)

    ax.legend(loc='best')

    plt.savefig(test_dir + '/tsne_visualization_both_sensors.png')
    plt.clf()


    # we need to subtract half a voxel size from the vertices to get to the voxel points 
    # since the marching cubes algorithm of open3d thinks that the tsdf voxel vertices are
    # always located at the mid point between the metric space resolution i.e. if we have a tsdf
    # grid of shape 2,2,2 and a voxel size of 1, the marching cubes algorithm will generate a surface at 0.5, 0.5, 0.5
    # to 1.5, 1.5, 1.5.

    # colorize the per sensor meshes with the color that is inter sensor consistent
    for sensor_ in features.keys():
        # load mesh using trimesh since trimesh can save RGBA colors per vertex
        mesh = trimesh.load_mesh(mesh_paths[sensor_])

        mesh.visual.vertex_colors = color_per_sensor[sensor_]
        mesh.remove_degenerate_faces()
        mesh.export(test_dir + '/inter_feature_visualization_' + sensor_ + '.ply')
        os.system('rm ' + mesh_paths[sensor_]) # clean up

            



# def visualize_features(tsdfs, mesh_paths, features, test_dir, mask, voxel_size):
#     # how should I go from 2 features to rgb colors. The most obvious is: get the maximum norm of the features
#     # in both grids and normalize with this value. This means later that the vector length is the alpha in 
#     # (RGBA). The phase is then the input to the cmap.

#     # determine the 

#     # but first we need to use tsne to make our feature space 2-dimensional 

#     # load feature data into data matrix X
#     X = None
#     for k, sensor_ in enumerate(features.keys()):
#         if k == 0:
#             X = features[sensor_][mask[sensor_]]
#         else:
#             X = np.concatenate((X, features[sensor_][mask[sensor_]]), axis=0)

#     print('Input shape to TSNE: ', X.shape)
#     # We want to get TSNE embedding with 2 dimensions
#     # n_components = 2
#     # tsne = TSNE(n_components)
#     # tsne_result = tsne.fit_transform(X[:10, :])
#     # tsne_result.shape


#     tsne = TSNE(
#         perplexity=3,
#         learning_rate='auto',
#         metric="euclidean",
#         # callbacks=ErrorLogger(),
#         n_jobs=-1, # use all cores
#         negative_gradient_method='fft',
#         random_state=42,
#     )
#     # compute tsne embedding of both sensors to compare inter sensor variation of features
#     tsne_result = tsne.fit(X)
#     # # save tsne data for further analysis
#     # np.savetxt(test_dir + '/tsne_embedding.txt', tsne_result)
#     # tsne_result = np.loadtxt(test_dir + '/tsne_embedding.txt')
#     print('TSNE output shape: ', tsne_result.shape)


#     # compute per sensor tsne embeddings to see intra sensor variation of features
#     tsne_per_sensor = dict()
#     for sensor_ in features.keys():
#         X = features[sensor_][mask[sensor_]]
        
#         if np.std(X[:, 0]) == 0 and np.std(X[:, 1]) == 0:
#             np.savetxt(test_dir + '/feature_embedding_' + sensor_ + '.txt', X)
#             continue
#         tsne_per_sensor[sensor_] = tsne.fit(X)

#     # get colors for the tsne embedding of both sensors
#     colors_both_sensors = get_colors_from_tsne_embedding(tsne_result)

#     # get colors for the tsne embedding that is sensor specific
#     colors_per_sensor = dict()
#     for key in tsne_per_sensor.keys():
#         colors_per_sensor[key] = get_colors_from_tsne_embedding(tsne_per_sensor[key])

#     # split the colors to each sensor specific grid
#     color_grid_per_sensor = dict()
#     color_grid_both_sensors = dict()
#     tsne_both_sensors = dict()

#     w, h, d, n = features[list(features.keys())[0]].shape

#     # put colors from the inter sensor study in color grids and separete the per sensor tsne embeddings
#     break_point = features[list(features.keys())[0]][mask[list(features.keys())[0]]].reshape(-1, n).shape[0]
#     for k, sensor_ in enumerate(features.keys()):
#         color_grid_both_sensors[sensor_] = np.ones((w, h, d, 4)) # vertices that are
#         # not assigned a color are white (due to speed up requirement of not embedding
#         # the entire feature grid as tsne but only close to the 0-crossing)
        
#         if k == 0:
#             color = colors_both_sensors[:break_point]
#             tsne = tsne_result[:break_point]
#         else:
#             color = colors_both_sensors[break_point:]
#             tsne = tsne_result[break_point:]

#         color_grid_both_sensors[sensor_][mask[sensor_], :] = color
        
#         tsne_both_sensors[sensor_] = tsne

#     # put colors from the intra sensor study in color grids
#     for sensor_ in tsne_per_sensor.keys():
#         color_grid_per_sensor[sensor_] = np.ones((w, h, d, 4))
#         color_grid_per_sensor[sensor_][mask[sensor_], :] = colors_per_sensor[sensor_] 

#     # plot tsne data with sensor labels
#     # make a mapping from category to your favourite colors and labels
#     sensor_to_color = {list(features.keys())[0]: 'lightgreen'}
#     if len(list(features.keys())) > 1:
#         sensor_to_color[list(features.keys())[1]] = 'darkgreen' 

#     # plot each category with a distinct label
#     fig, ax = plt.subplots(1,1)
#     for sensor_, color in sensor_to_color.items():
#         data = tsne_both_sensors[sensor_]
#         ax.plot(data[:, 0], data[:, 1], 'o', 
#                 color=color, label=sensor_)

#     ax.legend(loc='best')

#     plt.savefig(test_dir + '/tsne_visualization_both_sensors.png')
#     plt.clf()

#     for sensor_ in tsne_per_sensor.keys():
#         fig, ax = plt.subplots(1,1)
#         data = tsne_per_sensor[sensor_]
#         ax.plot(data[:, 0], data[:, 1], 'o', 
#                 color=color, label=sensor_)
#         ax.legend(loc='best')
#         plt.savefig(test_dir + '/tsne_visualization_' + sensor_ + '.png')
#         plt.clf()




#     # we need to subtract half a voxel size from the vertices to get to the voxel points 
#     # since the marching cubes algorithm of open3d thinks that the tsdf voxel vertices are
#     # always located at the mid point between the metric space resolution i.e. if we have a tsdf
#     # grid of shape 2,2,2 and a voxel size of 1, the marching cubes algorithm will generate a surface at 0.5, 0.5, 0.5
#     # to 1.5, 1.5, 1.5.

#     # colorize the per sensor meshes with the color that is inter sensor consistent
#     for sensor_ in features.keys():
#         # load mesh using trimesh since trimesh can save RGBA colors per vertex
#         mesh = trimesh.load_mesh(mesh_paths[sensor_])
#         # read vertices from mesh
#         vertices = mesh.vertices
#         voxel_points = np.round(np.asarray(vertices) * 1/voxel_size - voxel_size/2).astype(int)
#         a = color_grid_both_sensors[sensor_][voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :]
#         mesh.visual.vertex_colors = a
#         mesh.export(test_dir + '/inter_feature_visualization_' + sensor_ + '.ply')

#     # colorize the per sensor meshes with the color that is intra sensor consistent
#     for sensor_ in tsne_per_sensor.keys():
#         # load mesh using trimesh since trimesh can save RGBA colors per vertex
#         mesh = trimesh.load_mesh(mesh_paths[sensor_])
#         # read vertices from mesh
#         vertices = mesh.vertices
#         voxel_points = np.round(np.asarray(vertices) * 1/voxel_size - voxel_size/2).astype(int)
#         a = color_grid_per_sensor[sensor_][voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], :]
#         mesh.visual.vertex_colors = a
#         mesh.export(test_dir + '/intra_feature_visualization_' + sensor_ + '.ply')


def get_colors_from_tsne_embedding(tsne_result):
    cmap = plt.get_cmap("hsv") # it is important to use a cyclic color map since we want the color map to be 
    # continuous when the vector loops around

    # find the maximum length vector and normalize with this value
    # compute norm over all data points
    norm = np.linalg.norm(tsne_result, axis=1)
    max_norm = np.amax(norm) 
    norm = norm / max_norm # here we have the alpha value for all points

    # compute the phase of all points
    phase = np.angle(tsne_result[:, 0] + 1j * tsne_result[:, 1])
    # the output from phase is a number between -pi to pi. This
    # needs to be transformed to 0-1 for the cmap
    phase = phase / math.pi # range -1 to 1
    phase = (phase + 1)/2 # range 0-1

    # get the color from the colormap
    colors = cmap(phase)

    colors = np.concatenate((colors[:, :3], np.expand_dims(norm, -1)), axis=1)
    return colors
