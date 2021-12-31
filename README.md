# SenFuNet

This is the official source code of the research paper [**Learning Online Multi-Sensor Depth Fusion**](add_link_here).

Many hand-held or mixed reality devices typically use a single sensor for 3D reconstruction although they often comprise multiple sensors. Multi-sensor depth fusion is able to substantially improve the robustness and accuracy of 3D reconstruction methods, but existing techniques are not robust enough to handle sensors which operate in different domains. To this end, we introduce SenFuNet, a depth fusion approach that learns sensor-specific noise and outlier statistics and combines the data streams of depth frames from different sensors in an online fashion. Our method fuses multi-sensor depth streams regardless of synchronization and calibration and generalizes well with little training data.

### Prerequisites
The code has been tested with Python 3.8.5

## Installation

1. Clone the repository and submodules to your local directory: <pre><code>git clone --recursive https://github.com/tfy14esa/SenFuNet.git</code></pre>
2. Create a python virtual environment: <pre><code>python -m venv senfunet_env</code></pre>
3. Activate the virtual environment: <pre><code>source senfunet_env/bin/activate</code></pre>
4. Install the dependencies: <pre><code>pip install --ignore-installed -r requirements.txt</code></pre>
5. You may need to add the venv.patch file to include it and run bash venv.patch and place this file in the bin of the virtual environment. Ask Samuel if needed.
 
### Data Preparation
Put script of how to generate the MVS depth data here for scene3d and corbs, but not much more info.

For replica: put some data on biwimaster01
/usr/biwimaster01/data-biwi01/$USERNAME perhaps. Wait for Kris's answer.

### Training
Describe 

List the execution script in pure python form or as a batch job.
### Testing
Describe names of models that are available. Create folder with trained models with appropriate names. List the execution script in pure python form or as a batch job.

### Config Options
List some common config options that the use may want to change.