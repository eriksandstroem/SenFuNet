import argparse
import os

import numpy as np

import matplotlib.pyplot as plt

import cv2


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Script for creating a video of the depth."
    )

    parser.add_argument("--scene", required=True)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    return vars(args)


# From Johannes Schoenberger code.
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)

    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def get_depth(sensor, scene, trajectory, dataset):
    if dataset == "replica":
        input_dir = (
            "/cluster/work/cvl/esandstroem/data/replica/manual/"
            + scene
            + "/"
            + trajectory
            + "/"
            + sensor
        )
    else:
        if sensor == "tof":
            # corbs
            # input_dir = '/cluster/work/cvl/esandstroem/data/corbs/human/data/H1_pre_registereddata/depth'
            # scene3d
            input_dir = (
                "/cluster/work/cvl/esandstroem/data/scene3d/copyroom/copyroom_png/depth"
            )
        else:
            # corbs
            # input_dir = '/cluster/work/cvl/esandstroem/data/corbs/human/colmap/dense/stereo/depth_maps'
            # scene3d
            input_dir = "/cluster/work/cvl/esandstroem/data/scene3d/copyroom/dense/stereo/depth_maps"

    # define output dir
    output_folder = "/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/videos/"
    output_folder += "depth/" + scene + "/" + sensor

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = os.listdir(input_dir)

    if dataset == "replica":
        images = sorted(images, key=lambda x: float(x[:-4]))
    else:
        if sensor == "tof":
            images = sorted(images, key=lambda x: float(x[:-4]))
        else:
            images = [x for x in images if x.endswith("geometric.bin")]
            images = sorted(images, key=lambda x: float(x[:-18]))

    for k, im in enumerate(images):
        # print(im)
        if dataset == "replica":
            im = cv2.imread(input_dir + "/" + im, -1)
        elif sensor == "tof":
            im = cv2.imread(input_dir + "/" + im, -1)
        else:
            im = read_array(input_dir + "/" + im)

        print(k)
        # cv2.imwrite(im, input_dir + '/' + im)
        print(output_folder)
        plt.imsave(
            output_folder + "/" + "%04d" % k + ".png",
            np.asarray(im),
            vmin=0,
            vmax=5,
            dpi=1,
        )

    # vmin=0, vmax=25000
    # if k > 100:
    # break

    # create video of the rendered images
    os.chdir(output_folder)
    os.system(
        "ffmpeg -framerate 15 -i %04d.png -vcodec libx264 -preset veryslow -c:a libmp3lame -r 15 -crf 25 -pix_fmt yuv420p "
        + "/".join(output_folder.split("/")[:-1])
        + ".mp4"
    )

    # remove the images folder
    os.system("rm -r " + output_folder)


if __name__ == "__main__":

    # parse commandline arguments
    args = arg_parse()

    get_depth(args["sensor"], args["scene"], args["trajectory"], args["dataset"])
