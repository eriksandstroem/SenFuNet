import os
import argparse

import numpy as np

from data3d.utils.associate import associate
from pyquaternion import Quaternion

def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path')
    parser.add_argument('--colmap_path')

    # dataset options
    parser.add_argument('--sequence_id', default='H1')

    # camera options 
    parser.add_argument('--fx', default=468.60, type=float)
    parser.add_argument('--fy', default=468.61, type=float)
    parser.add_argument('--cx', default=318.27, type=float)
    parser.add_argument('--cy', default=243.99, type=float)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)

    args = parser.parse_args()
    return vars(args)

def main(args):
    
    IMAGE_PATH = os.path.join(args['colmap_path'], 'images')
    SPARSE_PATH = os.path.join(args['colmap_path'], 'sparse')
    DENSE_PATH = os.path.join(args['colmap_path'], 'dense')
    STEREO_PATH = os.path.join(DENSE_PATH, 'stereo')
    # setup colmap workspace
    if not os.path.exists(args['colmap_path']):
        os.makedirs(args['colmap_path'])
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
    if not os.path.exists(SPARSE_PATH):
        os.makedirs(SPARSE_PATH)
    if not os.path.exists(DENSE_PATH):
        os.makedirs(DENSE_PATH)
    if not os.path.exists(STEREO_PATH):
        os.makedirs(STEREO_PATH)

    # write camera file
    with open(os.path.join(SPARSE_PATH, 'cameras.txt'), 'w') as file:
        file.write('1 PINHOLE {} {} {} {} {} {}'.format(args['width'], args['height'], args['fx'], args['fy'], args['cx'], args['cy']))

    # write points file
    with open(os.path.join(SPARSE_PATH, 'points3D.txt'), 'w') as file:
        pass

    # copy images as build images file

    # build dictionary timestamp -> path
    timestamp_mapping = {}
    with open(os.path.join(args['source_path'], '{}_pre_registereddata/rgb.txt'.format(args['sequence_id'])), 'r') as file:
        for line in file:

            # skip comments
            if line[0] == '#':
                continue
            
            line = line.rstrip()
            timestamp, file_path = line.split(' ')
            timestamp_mapping[float(timestamp)] = file_path.replace('\\', '/')
    
    # iterate through trajectory
    poses = {}
    
    with open(os.path.join(args['source_path'], '{}_Trajectory/groundtruth.txt'.format(args['sequence_id'])), 'r') as file:
        for line in file:
            # skip comments
            if line[0] == '#':
                continue
            
            # parse and reformat data
            line = line.rstrip()
            elem = line.split(' ')
            timestamp = float(elem[0])

            # transform pose
            rotation = [float(e) for e in elem[4:]]
            rotation = Quaternion(rotation[-1], rotation[0], rotation[1], rotation[2])
            rotation = rotation.rotation_matrix
            translation = [float(e) for e in elem[1:4]]
            
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotation
            extrinsics[:3, 3] = translation

            # # invert for colmap
            extrinsics = np.linalg.inv(extrinsics)

            rotation = Quaternion(matrix=extrinsics[:3, :3])
            rotation = [rotation.elements[0], rotation.elements[1], rotation.elements[2], rotation.elements[3]]
            translation = list(extrinsics[:3, 3])

            pose = rotation + translation
            pose = [str(p) for p in pose]
            pose = " ".join(pose)

            # check correct length of pose
            assert len(pose.split(' ')) == 7

            poses[timestamp] = pose
    
    matches = associate(poses, timestamp_mapping, offset=0.0, max_difference=0.02)

    # write and copy images
    with open(os.path.join(SPARSE_PATH, 'images.txt'), 'w') as file, open(os.path.join(STEREO_PATH, 'patch-match.cfg'), 'w') as cfg:
        for i, (t_p, t_f) in enumerate(matches):
            
            # get data
            try:
                pose = poses[t_p]
                file_path = timestamp_mapping[t_f]
            except KeyError:
                continue
            
            image_line = '{} '.format(i + 1) + pose + ' {} '.format(1) + file_path.replace('rgb/', '') + '\n' + '\n'
            file.write(image_line)

            source_image = os.path.join(args['source_path'], '{}_pre_registereddata'.format(args['sequence_id']), file_path)
            target_image = os.path.join(IMAGE_PATH, file_path.replace('rgb/', ''))
            os.system('cp -p {} {}'.format(source_image, target_image))

            # write patch match config file
            cfg.write(file_path.replace('rgb/', '') + '\n')
            
            # get source images
            start_indx = max(0, i - 10)
            end_indx = min(len(matches), i + 10)
            source_images = []
            for j in range(start_indx, end_indx):
                if j == i:
                    continue
                source_images.append(timestamp_mapping[matches[j][1]].replace('rgb/', ''))
            
            # source_images = ", ".join(source_images)  
            # cfg.write('{}\n'.format(source_images))
            cfg.write('__auto__, 20\n')



if __name__ == '__main__':
    args = arg_parse()
    main(args)