#!/bin/bash

#BSUB -W 24:00
#BSUB -R "rusage[mem=2500, ngpus_excl_p=1]"
#BSUB -n 20
#BSUB -oo /cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/scripts/log

# call your calculation executable, redirect output BSUB -R "select[model!=EPYC_7742]" BSUB -R "select[gpu_mtotal0<=15000]"
export PATH=/cluster/project/cvl/esandstroem/virtual_envs/multisensor_env_python_gpu_3.8.5/bin:$PATH
export PYTHONPATH=/cluster/project/cvl/esandstroem/virtual_envs/multisensor_env_python_gpu_3.8.5/lib/python3.8/site-packages:$PYTHONPATH

python -u /cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/create_video.py --config /cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/configs/fusion/replica_euler.yaml --scene office_0
