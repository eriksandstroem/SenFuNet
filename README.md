# SenFuNet

This is the official source code of the research paper [**Learning Online Multi-Sensor Depth Fusion**](add_link_here).

![Architecture](https://github.com/tfy14esa/SenFuNet/blob/main/images/architecture.png)

Many hand-held or mixed reality devices typically use a single sensor for 3D reconstruction although they often comprise multiple sensors. Multi-sensor depth fusion is able to substantially improve the robustness and accuracy of 3D reconstruction methods, but existing techniques are not robust enough to handle sensors which operate in different domains. To this end, we introduce SenFuNet, a depth fusion approach that learns sensor-specific noise and outlier statistics and combines the data streams of depth frames from different sensors in an online fashion. Our method fuses multi-sensor depth streams regardless of synchronization and calibration and generalizes well with little training data.

## Prerequisites
The code has been tested with Python 3.8.5

## Installation

1. Clone the repository and submodules to your local directory: <pre><code>git clone --recursive https://github.com/tfy14esa/SenFuNet.git</code></pre>
2. Create a python virtual environment: <pre><code>python -m venv senfunet_env</code></pre>
3. Activate the virtual environment: <pre><code>source senfunet_env/bin/activate</code></pre>
4. Install the dependencies: <pre><code>pip install --ignore-installed -r requirements.txt</code></pre>
5. You may need to add the venv.patch file to include it and run bash venv.patch and place this file in the bin of the virtual environment. Ask Samuel if needed.
 
## Data Preparation
For replica: put some data on biwimaster01
/usr/biwimaster01/data-biwi01/$USERNAME perhaps. Wait for Kris's answer.

Prepare 2D data
Replica stuff

Prepare ground truth 3D data
Note that the user needs the tsdf GT grids and gt meshes. Specify paths in F-score eval config.py


Put script of how to generate the MVS depth data here for scene3d and corbs, but not much more info.


## Training
To train SenFuNet, execute the script:

<pre><code>python train_fusion.py --ROOT_FOLDER/configs/fusion/CONFIG.yaml</code></pre>

where ROOT_FOLDER is the path to the SenFuNet folder and CONFIG is a config file.

To train using the LSF job scheduler, use one of the submission scripts located in the "scripts" folder. To submit a job, run:
<pre><code>bsub > SUBMISSION_SCRIPT.sh</code></pre>
where SUBMISSION_SCRIPT is the appropriate submission script. Note that you need to specify the appropriate paths on your system in the submission scripts.

The following paths need to be specified in the CONFIG.yaml file:
1. SETTINGS.experiment_path -> path where the logging is done and the models are saved.
2. DATA.root_dir -> Path to data folder
3. train_scene_list
4. val_scene_list
5. test_scene_list

For the Replica dataset, in addition, specify the following paths. Only the Replica dataset can be trained and tested with routing.

6. routing_stereo_model_path -> path to psmnet stereo routing model (only used when training with a routing network)
7. routing_tof_model_path -> path to tof routing model (only used when training with a routing network)
8. routing_tof_2_model_path -> same path as "routing_tof_model_path". Used when doing collaborative reconstruction.

The routing models are located at ROOT_FOLDER/models/routing.

## Testing
To test SenFuNet, execute the script:

<pre><code>python test_fusion.py --ROOT_FOLDER/configs/fusion/CONFIG.yaml</code></pre>

where ROOT_FOLDER is the path to the SenFuNet folder and CONFIG is a config file.

To train using the LSF job scheduler, use one of the submission scripts located in the "scripts" folder. To submit a job, run:
<pre><code>bsub > SUBMISSION_SCRIPT.sh</code></pre>
where SUBMISSION_SCRIPT is the appropriate submission script. Note that you need to specify the appropriate paths on your system in the submission scripts.

The following paths need to be specified in the CONFIG.yaml file:
1. SETTINGS.experiment_path -> path where the logging is done and the models are saved.
2. DATA.root_dir -> Path to data folder
3. train_scene_list
4. val_scene_list
5. test_scene_list
6. fusion_model_path -> path to model to be tested. If routing was used during training, this model will include the routing network parameters as well.

The fusion models are located at ROOT_FOLDER/models/fusion.

Note that the copy room test scene of the Scene3D dataset does not come with an accurate ground truth TSDF grid and thus only the F-score evaluation is accurate.

### Reproduce Results in Paper

List some common config options that the use may want to change.

Check what other people typically put in these kind of readmes.

## Train Routing Network



