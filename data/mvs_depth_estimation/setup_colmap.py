import os
import argparse

import numpy as np

from pyquaternion import Quaternion

def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', default='/scratch-second/esandstroem/opportunistic_3d_capture/data/scene3d', type=str)
    parser.add_argument('--colmap_path', default='/scratch-second/esandstroem/opportunistic_3d_capture/data/scene3d', type=str)

    # camera options 
    parser.add_argument('--fx', default=525.00, type=float)
    parser.add_argument('--fy', default=525.00, type=float)
    parser.add_argument('--cx', default=319.5, type=float)
    parser.add_argument('--cy', default=239.5, type=float)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)

    args = parser.parse_args()
    return vars(args)

def main(args):
    scenes = ['cactusgarden', 'lounge', 'copyroom']
    for scene in scenes:
        IMAGE_PATH = os.path.join(args['colmap_path'], scene, 'images')
        TRAJECTORY_PATH = os.path.join(args['colmap_path'], scene, scene + '_trajectory.log')
        SPARSE_PATH = os.path.join(args['colmap_path'], scene, 'sparse')
        DENSE_PATH = os.path.join(args['colmap_path'], scene, 'dense')
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
        
        poses = dict()
        # retrieve pose dictionary
        with open(TRAJECTORY_PATH, 'r') as file:

            for rgb_name in sorted(os.listdir(IMAGE_PATH)):
                # extract the camera extrinsics by reading 5 lines
                metadata = next(file)
      
                first = np.fromstring(next(file), count=4, sep=' ', dtype=float) #[:-1].split(' ')
                second = np.fromstring(next(file), count=4, sep=' ', dtype=float)
                third = np.fromstring(next(file), count=4, sep=' ', dtype=float)
                fourth = np.fromstring(next(file), count=4, sep=' ', dtype=float)

                extrinsics = np.zeros((4,4))
                extrinsics[0, :] = first
                extrinsics[1, :] = second
                extrinsics[2, :] = third
                extrinsics[3, :] = fourth

                # print(np.matmul(extrinsics[:3, :3] , np.transpose(extrinsics[:3, :3])))
                # invert for colmap
                extrinsics = np.linalg.inv(extrinsics)

                rotation = Quaternion(matrix=extrinsics[:3, :3], rtol=1e-04, atol=1e-04)
                rotation = [rotation.elements[0], rotation.elements[1], rotation.elements[2], rotation.elements[3]]
                translation = list(extrinsics[:3, 3])

                pose = rotation + translation
                pose = [str(p) for p in pose]
                pose = " ".join(pose)

                # check correct length of pose
                assert len(pose.split(' ')) == 7
                # print(rgb_name)
                poses[rgb_name] = pose

        # write and copy images
        with open(os.path.join(SPARSE_PATH, 'images.txt'), 'w') as file, open(os.path.join(STEREO_PATH, 'patch-match.cfg'), 'w') as cfg:
            
            for i, rgb_name in enumerate(sorted(os.listdir(IMAGE_PATH))):
                
                # add rgb name to patch-match.cfg file
                cfg.write(rgb_name + '\n')
                # limit the number of source images during reconstruction to 20 to reduce memory requirement
                cfg.write('__auto__, 20\n')
                # if specifying source images manually
                # get source images
                # start_indx = max(0, i - 10)
                # end_indx = min(len(matches), i + 10)
                # source_images = []
                # for j in range(start_indx, end_indx):
                #     if j == i:
                #         continue
                #     source_images.append(timestamp_mapping[matches[j][1]].replace('rgb/', ''))
                # source_images = ", ".join(source_images)  
                # cfg.write('{}\n'.format(source_images))
                
                # retrieve pose for the rgb frame
                image_line = '{} '.format(i + 1) + poses[rgb_name] + ' {} '.format(1) + rgb_name + '\n' + '\n'
                file.write(image_line)


if __name__ == '__main__':
    args = arg_parse()
    main(args)