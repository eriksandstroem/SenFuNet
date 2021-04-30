#!/bin/bash

#SBATCH --output=log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

# call your calculation executable, redirect output
export PATH=/home/esandstroem/scratch/venvs/routedfusion_env/bin:$PATH

python /home/esandstroem/scratch-second/opportunistic_3d_capture/Erik_3D_Reconstruction_Project/src/RoutedFusion/test_routing.py --config /home/esandstroem/scratch-second/opportunistic_3d_capture/Erik_3D_Reconstruction_Project/src/RoutedFusion/configs/routing/replica.yaml








