# SenFuNet
This is the official source code of the research paper [**Learning Online Multi-Sensor Depth Fusion**](https://arxiv.org/abs/2204.03353).

![Architecture](https://github.com/tfy14esa/SenFuNet/blob/main/images/architecture.png)

Many hand-held or mixed reality devices typically use a single sensor for 3D reconstruction although they often comprise multiple sensors. Multi-sensor depth fusion is able to substantially improve the robustness and accuracy of 3D reconstruction methods, but existing techniques are not robust enough to handle sensors which operate in different domains. To this end, we introduce SenFuNet, a depth fusion approach that learns sensor-specific noise and outlier statistics and combines the data streams of depth frames from different sensors in an online fashion. Our method fuses multi-sensor depth streams regardless of synchronization and calibration and generalizes well with little training data.

Please check out [**this**](https://youtu.be/woA8FU05AM0) video which describes our method and shows the most important results.

This repository provides all necessary steps to reproduce the results in the paper including data preparation and trained models that can be tested by the user directly.

## Prerequisites
The code has been tested with Python 3.8.5

## Installation

1. Clone the repository and submodules to your local directory: <pre><code>git clone --recursive https://github.com/tfy14esa/SenFuNet.git</code></pre> In the following, we denote the local path to the SenFuNet codebase as "ROOT_FOLDER".
2. Create a python virtual environment: <pre><code>python -m venv senfunet_env</code></pre>
3. Activate the virtual environment: <pre><code>source senfunet_env/bin/activate</code></pre>
4. Open the requirements.txt and change the path to the submodule evaluate-3d-reconstruction according to your file structure. Set the variable "ground_truth_data_base" to the directory where you store the ground truth .ply meshes (see below under the header "Ground Truth Meshes"). This variable is located in the config.py file of the evaluate_3d_reconstruction library located in the "deps" folder.
5. Open the requirements.txt and replace the Open3D entry with a version of your choice. Install version 0.13.0 or newer (this has not been tested). Replace the entry with e.g. "open3d==0.13.0". A local version of the Open3D library was used when developing the project which is the reason why this entry needs to be replaced.
5. Install the dependencies: <pre><code>pip install -r requirements.txt</code></pre>
6. Note that the project uses "Weights and Biases" for logging during training. If you want to train your own models, therefore create your own account at [**Weights and Biases**](https://wandb.ai/site).
<!---
6. TODO: You may need to add the venv.patch file to include it and run bash venv.patch and place this file in the bin of the virtual environment. Ask Samuel if needed. Check riot how the file is executed.
-->
 
## Data Preparation
We provide three separate datasets that can be used with SenFuNet. The download links and instructions are provided below. 

### Ground Truth Meshes
Download the ground truth meshes used for F-score evaluation [**here**](https://data.vision.ee.ethz.ch/esandstroem/gt_meshes.tar).

Note: be sure to set the variable "ground_truth_data_base" to the directory where you store the ground truth .ply meshes.
This variable is located in the config.py file of the evaluate_3d_reconstruction library located in the "deps" folder. This step has to be performed before installing the module via pip.

While not needed for 
this codebase, there is also the option to set the path to the tranformation folder where transformation matrices 
are stored which aligns the ground truth mesh and the predicted mesh before F-score evaluation.

### Replica
Train Dataset: [**room 0**](https://data.vision.ee.ethz.ch/esandstroem/replica/room_0.tar), 
[**room 2**](https://data.vision.ee.ethz.ch/esandstroem/replica/room_2.tar), [**office 3**](https://data.vision.ee.ethz.ch/esandstroem/replica/office_3.tar), [**office 1**](https://data.vision.ee.ethz.ch/esandstroem/replica/office_1.tar), [**apartment 1**](https://data.vision.ee.ethz.ch/esandstroem/replica/apartment_1.tar), [**frl apartment 0**](https://data.vision.ee.ethz.ch/esandstroem/replica/frl_apartment_0.tar)

Validation Dataset: [**frl apartment 1**](https://data.vision.ee.ethz.ch/esandstroem/replica/frl_apartment_1.tar)

Test Dataset: [**office 0**](https://data.vision.ee.ethz.ch/esandstroem/replica/office_0.tar), [**hotel 0**](https://data.vision.ee.ethz.ch/esandstroem/replica/hotel_0.tar), [**office 4**](https://data.vision.ee.ethz.ch/esandstroem/replica/office_4.tar)

Important: Store your dataset at a location specified in the config variable <pre><code>DATA.root_dir</code></pre> of the <pre><code>ROOT_FOLDER/configs/fusion/replica.yaml</code></pre> config file.

Quick Setup: In the event that you don't want to donwload the full dataset at first, only download the office 0 scene and use this for training, validation and testing. In that case, please change the paths of the variables <pre><code>DATA.train_scene_list, DATA.val_scene_list, DATA.test_scene_list</code></pre> listed in the config file <pre><code>ROOT_FOLDER/configs/fusion/replica.yaml</code></pre>.

### CoRBS
The CoRBS dataset can be downloaded [**here**](http://corbs.dfki.uni-kl.de/). The CoRBS dataset does not include multi-view stereo (MVS) depth maps nor the ground truth signed distance grids (SDF). We provide these separately. 

Train and Validation Dataset: Download the D1 trajectory of the desk scene. The corresponding MVS depth maps and ground truth SDF grid are available [**here**](https://data.vision.ee.ethz.ch/esandstroem/corbs/desk.tar).

Test Dataset: Download the H1 trajectory of the human scene. The corresponding MVS depth maps and ground truth SDF grid are available [**here**](https://data.vision.ee.ethz.ch/esandstroem/corbs/human.tar).

Prepare the data such that the paths to the depth sensors and camera matrices are provided in the files <pre><code>ROOT_FOLDER/lists/human.txt</code></pre> and <pre><code>ROOT_FOLDER/lists/desk.txt</code></pre> For each row the entries are separated by spaces and structured as follows: <pre><code>PATH_TO_MVS_DEPTH PATH_TO_FOLDER_WITH_TOF_DEPTH PATH_TO_RGB_TIMESTAMP_FILE PATH_TO_TOF_TIMESTAMP_FILE CAMERA_MATRICES</code></pre>

### Scene3D
Download the stonewall and copy room scenes of the Scene3D dataset available [**here**](https://www.qianyi.info/scenedata.html). Next, use the script <pre><code>ROOT_FOLDER/data/save_every_tenth_frame.py</code></pre> to save every tenth sample (we only integrate every tenth frame).

Next, download the MVS depth sensor for both scenes and ground truth SDF grid for the stonewall training scene. We were not able to construct a good ground truth SDF grid for the copy room scene thus only the F-score evaluation is accurate at test time. Download links: [**stonewall**](https://data.vision.ee.ethz.ch/esandstroem/scene3d/stonewall.tar), [**copy room**](https://data.vision.ee.ethz.ch/esandstroem/scene3d/copyroom.tar).

Arrange the data such that the paths listed in the corresponding <pre><code>ROOT_FOLDER/lists/*.txt</code></pre> file match your folder structure. For each for, the entries are separated by spaces and structured as follows: <pre><code>PATH_TO_RGB PATH_TO_TOF_DEPTH PATH_TO_MVS_DEPTH CAMERA_MATRICES</code></pre>.

### Generate Multi-View Stereo Depth
In the event that you want to reproduce or generate your own MVS depth sensors, we provide the scripts for this. These are available in the folder <pre><code>ROOT_FOLDER/data/mvs_depth_estimation</code></pre>. First use the script <pre><code>setup_colmap.py</code></pre> and then the script <pre><code>reconstruct_colmap_slurm_SCENE.sh</code></pre> to use to generate the MVS depth maps. For information, we refer to the [**colmap**](https://colmap.github.io/faq.html) documentation.

## Training
To train SenFuNet, execute the script:

<pre><code>python train_fusion.py --ROOT_FOLDER/configs/fusion/CONFIG.yaml</code></pre>

where CONFIG is a config file.

To train using the LSF job scheduler, use one of the submission scripts located in the "scripts" folder. To submit a job, run:
<pre><code>bsub > SUBMISSION_SCRIPT.sh</code></pre>
where SUBMISSION_SCRIPT is the appropriate submission script. Note that you need to specify the appropriate paths on your system in the submission scripts.

The following paths need to be specified in the CONFIG.yaml file:
1. SETTINGS.experiment_path -> path where the logging is done and the models are saved
2. DATA.root_dir -> Path to data folder
3. DATA.train_scene_list -> specifies data used during training
4. DATA.val_scene_list -> specifies data used during validation
5. DATA.test_scene_list -> specifies data used during testing

For the Replica dataset, in addition, specify the following paths. Note: only the Replica dataset can be trained and tested with depth denoising. For the remainder of this guide, we refer to the denoising network as the routing network which is the term used in the original paper [**RoutedFusion**](https://www.microsoft.com/en-us/research/uploads/prod/2020/06/RoutedFusion.pdf).

6. TRAINING.routing_stereo_model_path -> path to psmnet stereo routing model (only used when training with a routing network)
7. TRAINING.routing_tof_model_path -> path to tof routing model (only used when training with a routing network)
8. TRAINING.routing_tof_2_model_path -> same path as "routing_tof_model_path". Used when doing multi-agent reconstruction.

The routing models are located at ROOT_FOLDER/models/routing.

## Testing
To test SenFuNet, execute the script:

<pre><code>python test_fusion.py --ROOT_FOLDER/configs/fusion/CONFIG.yaml</code></pre>

where CONFIG is a config file.

To test using the LSF job scheduler, use one of the submission scripts located in the "scripts" folder. To submit a job, run:
<pre><code>bsub > SUBMISSION_SCRIPT.sh</code></pre>
where SUBMISSION_SCRIPT is the appropriate submission script. Note that you need to specify the appropriate paths on your system in the submission scripts.

The following paths need to be specified in the CONFIG.yaml file:
1. SETTINGS.experiment_path -> path where the logging is done and the models are saved.
2. DATA.root_dir -> Path to data folder
3. DATA.train_scene_list -> specifies data used during training
4. DATA.val_scene_list -> specifies data used during validation
5. DATA.test_scene_list -> specifies data used during testing
6. TESTING.fusion_model_path -> path to model to be tested. If routing was used during training, this model will include the routing network parameters as well.

The fusion models are located at ROOT_FOLDER/models/fusion.

### Configs to Reproduce Results in Paper
Note: The F-scores reported in the paper are computed using meshes which are produced using the marching cubes implementation of Open3D. Gaining access to this required some changes of the C++ source code and to make installation simpler, we resort to the skimage implementation. Only minor numerical differences exist between the two implementations.
 
**Replica**
1. ToF+PSMNet without routing. Use the config file located at ROOT_FOLDER/configs/fusion/replica.yaml
2. ToF+PSMNet with routing. Modify the config file ROOT_FOLDER/configs/fusion/replica.yaml as follows:
<pre><code>ROUTING.do: True
TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/tof_psmnet_routing/model/best.pth.tar</code></pre> 
3. SGM+PSMNet without routing. Modify the config file ROOT_FOLDER/configs/fusion/replica.yaml as follows:
<pre><code>TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/sgm_psmnet/model/best.pth.tar
DATA.input: [sgm_stereo, stereo]</code></pre> 
4. SGM+PSMNet with routing. Modify the config file ROOT_FOLDER/configs/fusion/replica.yaml as follows:
<pre><code>ROUTING.do: True
TESTING.fusion_model_path: ROOT_FOLDER/models/fusion/sgm_psmnet_routing/model/best.pth.tar
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
In the event that you want to train your own routing network, use the following guide to train and test the routing network.

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
3. DATA.train_scene_list -> specifies data used during training
4. DATA.val_scene_list -> specifies data used during validation
5. DATA.test_scene_list -> specifies data used during testing

### Testing
Execute the following command to test your routing network:

<pre><code>python test_routing.py --ROOT_FOLDER/configs/routing/replica.yaml</code></pre>

where ROOT_FOLDER is the path to the SenFuNet folder.

In addition to specifying the paths for training, this requires the model path variable <pre><code>TESTING.fusion_model_path</code></pre> to be specified in the replica.yaml config file.

The test script creates the output 16-bit depth images from the routing network but does not evaluate the depth accuracy. Quantitative evaluation of the depth maps needs to be done as a post processing step.

<!---
I can create the depth evaluation code as a separete python library and link here.
-->

## Baseline Methods
In the event that you want to reproduce our baseline results on the Replica dataset, follow the steps outlined below.

### Early Fusion
The early fusion baseline is only applicable to the Replica dataset since it requires ground truth depth maps for training. 
1. Select the appropriate sensor suite i.e. ToF+PSMNet or SGM+PSMNet. Change the config variable <pre><code>DATA.input</code></pre> appropriately.
2. Specify the path to the routing network using the config variable <pre><code>TESTING.routing_model_path</code></pre> For example, the early fusion routing network for ToF+PSMNet fusion is available at ROOT_FOLDER/models/routing/tof_psmnet/model/best.pth.tar
3. Set <pre><code>TESTING.use_outlier_filter: False
FILTERING_MODEL.model: 'tsdf_early_fusion'
ROUTING.do: True</code></pre>
6. Test using the test_fusion.py script with the config as input.
### TSDF Fusion
1. Select the appropriate sensor suite i.e. ToF+PSMNet or SGM+PSMNet. Change the config variable <pre><code>DATA.input</code></pre> appropriately.
2. Set <pre><code>TESTING.use_outlier_filter: False</code></pre>
3. Set <pre><code>FILTERING_MODEL.model: 'tsdf_middle_fusion'</code></pre>
4. Test using the test_fusion.py script with the config as input.
### RoutedFusion
1. Select the appropriate sensor suite i.e. ToF+PSMNet or SGM+PSMNet. Change the config variable <pre><code>DATA.input</code></pre> appropriately.
2. Specify the path to the fusion network using the config variable <pre><code>TESTING.fusion_model_path</code></pre> For example, the fusion network for ToF+PSMNet fusion without routing is available at ROOT_FOLDER/models/fusion/tof_psmnet_routedfusion/model/best.pth.tar
3. Set <pre><code>FUSION_MODEL.use_fusion_net: True 
FUSION_MODEL.extraction_strategy: 'trilinear_interpolation'
TESTING.use_outlier_filter: False
FILTERING_MODEL.model: 'routedfusion'</code></pre>
4. Note that to avoid excessive outliers from the RoutedFusion model, to produce the mesh from the predicted TSDF grid, we apply the nearest neighbor mask from the nearest neighbor extraction strategy and not the trilinear interpolation strategy. This requires loading a separate weight grid from a model of the same sensor suite but using nearest neighbor extraction. Set the name of the config variable <pre><code>TESTING.routedfusion_nn_model:</code></pre> Note that this requires testing and saving the grid of this model first.
5. Test using the test_fusion.py script with the config as input.
