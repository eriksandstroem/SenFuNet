# SenFuNet
TODO: What license should I use? Have I "cited" e.g. the colmap.py script correctly? It should be enough to include the header I suppose?
TODO: marching cubes issue! Check where the surface is drawn between open3d and skimage see comment on 1.5.
TODO: Check what other people typically put in these kind of readmes.

This is the official source code of the research paper [**Learning Online Multi-Sensor Depth Fusion**](add_link_here).

![Architecture](https://github.com/tfy14esa/SenFuNet/blob/main/images/architecture.png)

Many hand-held or mixed reality devices typically use a single sensor for 3D reconstruction although they often comprise multiple sensors. Multi-sensor depth fusion is able to substantially improve the robustness and accuracy of 3D reconstruction methods, but existing techniques are not robust enough to handle sensors which operate in different domains. To this end, we introduce SenFuNet, a depth fusion approach that learns sensor-specific noise and outlier statistics and combines the data streams of depth frames from different sensors in an online fashion. Our method fuses multi-sensor depth streams regardless of synchronization and calibration and generalizes well with little training data.

This repository provides all necessary steps to reproduce the results in the paper including data preparation and trained models that can be tested by the user directly.

## Prerequisites
The code has been tested with Python 3.8.5

## Installation

1. Clone the repository and submodules to your local directory: <pre><code>git clone --recursive https://github.com/tfy14esa/SenFuNet.git</code></pre>
2. Create a python virtual environment: <pre><code>python -m venv senfunet_env</code></pre>
3. Activate the virtual environment: <pre><code>source senfunet_env/bin/activate</code></pre>
4. Install the dependencies: <pre><code>pip install --ignore-installed -r requirements.txt</code></pre>
5. TODO: You may need to add the venv.patch file to include it and run bash venv.patch and place this file in the bin of the virtual environment. Ask Samuel if needed. Check riot how the file is executed.
 
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
3. DATA.train_scene_list
4. DATA.val_scene_list
5. DATA.test_scene_list

For the Replica dataset, in addition, specify the following paths. Only the Replica dataset can be trained and tested with routing.

6. TRAINING.routing_stereo_model_path -> path to psmnet stereo routing model (only used when training with a routing network)
7. TRAINING.routing_tof_model_path -> path to tof routing model (only used when training with a routing network)
8. TRAINING.routing_tof_2_model_path -> same path as "routing_tof_model_path". Used when doing multi-agent reconstruction.

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
3. DATA.train_scene_list
4. DATA.val_scene_list
5. DATA.test_scene_list
6. TESTING.fusion_model_path -> path to model to be tested. If routing was used during training, this model will include the routing network parameters as well.

The fusion models are located at ROOT_FOLDER/models/fusion.

Note that the copy room test scene of the Scene3D dataset does not come with an accurate ground truth TSDF grid and thus only the F-score evaluation is accurate.

### Configs to Reproduce Results in Paper

**Replica**
1. ToF+PSMNet without routing. Use the config file located at ROOT_FOLDER/configs/fusion/replica.yaml
2. ToF+PSMNet with routing. Modify the config file ROOT_FOLDER/configs/fusion/replica.yaml as follows:
<pre><code>ROUTING.do: True
TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/tof_psmnet_routing/model/best.pth.tar</code></pre> 
3. SGM+PSMNet without routing. Modify the config file ROOT_FOLDER/configs/fusion/replica.yaml as follows:
<pre><code>TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/sgm_psmnet/model/best.pth.tar
DATA.input: [sgm_stereo, stereo]</code></pre> 
4. SGM+PSMNet with routing. Modify the config file ROOT_FOLDER/configs/fusion/replica.yaml as follows:
<pre><code>TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/sgm_psmnet_routing/model/best.pth.tar
DATA.input: [sgm_stereo, stereo]</code></pre> 

**CoRBS**
1. ToF+MVS. Use the config file located at ROOT_FOLDER/configs/fusion/corbs.yaml.

**Scene3D**
1. ToF+MVS. Use the config file located at ROOT_FOLDER/configs/fusion/scene3d.yaml.
2. ToF+ToF (multi-agent reconstruction). Modify the config file ROOT_FOLDER/configs/fusion/scene3d.yaml as follows:
<pre><code>TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/tof_tof_scene3d_collab_rec/model/best.pth.tar
DATA.collaborative_reconstruction: True
DATA.input: [tof, tof_2]</code></pre> 

## Routing Network
In the event that you want to train your own routing network or simply reproduce the results, use the following guide to train and test the routing network.

### Training
Execute the following command to train a routing network:

<pre><code>python train_routing.py --ROOT_FOLDER/configs/routing/replica.yaml</code></pre>

where ROOT_FOLDER is the path to the SenFuNet folder.

To train using the LSF job scheduler, use one of the submission scripts located in the "scripts" folder. To submit a job, run:
<pre><code>bsub > SUBMISSION_SCRIPT.sh</code></pre>
where SUBMISSION_SCRIPT is the appropriate submission script. Note that you need to specify the appropriate paths on your system in the submission scripts.

The following paths need to be specified in the replica.yaml file:
1. SETTINGS.experiment_path -> path where the logging is done and the models are saved.
2. DATA.root_dir -> Path to data folder
3. DATA.train_scene_list
4. DATA.val_scene_list
5. DATA.test_scene_list

### Testing
Execute the following command to test your routing network:

<pre><code>python test_routing.py --ROOT_FOLDER/configs/routing/replica.yaml</code></pre>

where ROOT_FOLDER is the path to the SenFuNet folder.

In addition to specifying the paths for training, this requires the model path variable <pre><code>TESTING.fusion_model_path</code></pre> to be specified in the replica.yaml config file.

The test script creates the output 16-bit depth images from the routing network but does not evaluate the depth accuracy. Quantitative evaluation of the depth maps needs to be done as a post processing step.

TODO: I can create the depth evaluation code as a separete python library and link here.

## Baseline Methods
In the event that you want to reproduce our baseline results, we provide the trained models which can be tested directly by modifying the config files.

### Early Fusion

### TSDF Fusion

### RoutedFusion

