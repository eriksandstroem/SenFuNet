#!/bin/bash

# activate virtual environment
export PATH=/cluster/project/cvl/esandstroem/virtual_envs/multisensor_env_python_gpu_3.8.5/bin:$PATH
export PYTHONPATH=/cluster/project/cvl/esandstroem/virtual_envs/multisensor_env_python_gpu_3.8.5/lib/python3.8/site-packages:$PYTHONPATH

# path to config file
CONFIG_FILE=/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/configs/fusion/replica_euler.yaml

# function that retrieves the training or validation scenes depending on input mode
retrieve_scenes() {
		python -c "import retrieve_scenes; retrieve_scenes.retrieve_scenes('$1', '$2')"
}
	

# extract scenes
scene_string=$(retrieve_scenes $CONFIG_FILE 'train')
echo $scene_string

for SCENE in $scene_string
do
	echo $SCENE
	tar -I pigz -xf /cluster/work/cvl/$USER/data/replica/manual/$SCENE.tar -C ${TMPDIR}/
done

scene_string=$(retrieve_scenes $CONFIG_FILE 'val')

for SCENE in $scene_string
do
	echo $SCENE
	tar -I pigz -xf /cluster/work/cvl/$USER/data/replica/manual/$SCENE.tar -C ${TMPDIR}/
done


python -u /cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/train_fusion.py --config /cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/configs/fusion/replica_euler.yaml
