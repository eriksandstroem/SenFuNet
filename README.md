# SenFuNet

This is the official source code of the research paper [**Learning Online Multi-Sensor Depth Fusion**](add_link_here).

Many hand-held or mixed reality devices typically use a single sensor for 3D reconstruction although they often comprise multiple sensors. Multi-sensor depth fusion is able to substantially improve the robustness and accuracy of 3D reconstruction methods, but existing techniques are not robust enough to handle sensors which operate in different domains. To this end, we introduce SenFuNet, a depth fusion approach that learns sensor-specific noise and outlier statistics and combines the data streams of depth frames from different sensors in an online fashion. Our method fuses multi-sensor depth streams regardless of synchronization and calibration and generalizes well with little training data.

### Prerequisites
The code has been tested with Python 3.8.5

## Installation

1. Clone the repository to your local directory: <pre><code>git clone https://github.com/tfy14esa/evaluate_3d_reconstruction_lib.git</code></pre>
2. Open the file <pre><code>evaluate_3d_reconstruction/evaluate_3d_reconstruction/config.py</code></pre> and revise the paths to where you store the ground truth meshes and the transformation matrices. Then open the file <pre><code>evaluate_3d_reconstruction/evaluate_3d_reconstruction/evaluate_3d_reconstruction.py</code></pre> and revise the shebang at the top to point to the python executable of your virtual environment (this is useful if you want to execute the evaluation script directly from the command line (see below)).
2. Activate your virtual environment
3. Enter the root folder of the library: <pre><code>cd evaluate_3d_reconstruction</code></pre>
4. Install the library: <pre><code>pip install .</code></pre>
 
### Usage

The main function of the library is 
<pre><code>def run_evaluation(pred_ply, path_to_pred_ply, scene, transformation=None):
    """
    Calculates the F-score from a predicted mesh to a reference mesh. Generates
    a directory and fills this numerical and mesh results.

    Args:
        pred_ply: string object to denote the name of predicted mesh (as a .ply file)
        scene: string object to denote the scene name (a corresponding ground 
                    truth .ply file with the name "scene + .ply" needs to exist)
        path_to_pred_ply: string object to denote the full path to the pred_ply file
        transformation: boolean to denote if to use the available transformation matrix for
                    the ground truth mesh.
    Returns:
        None
    """
</code></pre>

The main function can be called in two principled ways:

1. As an executable directly from the command line as:
<pre><code>evaluate_3d_reconstruction.py pred_ply scene transformation</code></pre>
,or
<pre><code>evaluate_3d_reconstruction.py pred_ply scene</code></pre>
if the predicted and ground truth meshes are already aligned. To achieve this, run <pre><code>chmod +x evaluate_3d_reconstruction.py</code></pre> and export the path to the script in your bashrc-file i.e. add similar to the following to your bashrc: <pre><code># Export path to my python evaluate 3d reconstruction script
export PATH="/cluster/project/cvl/esandstroem/src/late_fusion_3dconv/deps/evaluate_3d_reconstruction/evaluate_3d_reconstruction:$PATH"</code></pre>

2. As a normal function in other python scripts. To achieve this, simply import the function using <pre><code>from evaluate_3d_reconstruction import run_evaluation</code></pre>
