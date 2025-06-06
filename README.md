# GaussianDreamer Demonstration (CSE252D)

-- By Bhavik Chandna and Isheta Bansal.

This repository presents a **CSE252D course demonstration** of **GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models**, originally developed by the [HuSVL group](https://github.com/hustvl/GaussianDreamer). It showcases how language prompts can be translated into 3D Gaussian splats using diffusion-based techniques.

📌 **Paper**: [CVPR 2024 Paper Link](https://arxiv.org/abs/2312.00768)  
📌 **Original Repository**: [hustvl/GaussianDreamer](https://github.com/hustvl/GaussianDreamer)

---

The actual repository of the paper has been fully integrated into [threestudio project](https://github.com/threestudio-project/threestudio) and has no clean code to understand each stage of the pipeline. This is the motivation of this repo. We have remove the threestudio dependancies and make the code simple and readable and integrated with gsplat. Check out the Colab Link(using my repo) for the demonstration.
If you dont want to understand code and want to run a simple colab demo, you can head over to [Colab Link (using Official Repo)](https://colab.research.google.com/drive/1PHdsF0PtqGb04FbcwAs_QIG_rNW7W4bh?usp=sharing)

GaussianDreamer embarks on the task of text to 3D generation through a two stage pipeline:

**3D Diffusion Prior**
A pre-trained 3D diffusion generator such as Shap E first conjures a vibrant point cloud or mesh from the text prompt, providing a pivotal geometric and colour prior for the scene. 
GitHub

**2D Diffusion Guided Gaussian Optimization**
The prior is converted to a field of 3D Gaussians whose positions, scales, orientations, colours, and opacities are refined with Score Distillation Sampling, using a 2D diffusion backbone like Stable Diffusion as a perceptual loss. Multi-view splats are rendered differentiably and updated until they comprehensively align with the prompt semantics. 

---

## How to run code?
## Environment Setup

To run the demonstration, clone this repository and configure your Python environment as follows:

```bash
python3 -m venv cse252d
source cse252d/bin/activate
git clone https://github.com/TheProParadox/gaussiandreamer-demo.git
cd gaussiandreamer-demo
pip install -r requirements.txt
```

## Code run
Use the following command to run the code: 

```bash
python main.py <prompt> <sample_proportion>
```
prompt = prompt as a string 
sample_proportion = value between 0.0 and 1.0 designating how many points to sample from the triangle mesh

Example: 
```bash
python main.py "a shark" 0.5
```

All output can be found on the right under the same folder.

There are 5 output files:

- "PointClouds.png" - visualization of generated point clouds
- "PointCloudsViews.png" - visualization of generated point clouds from different camera angles
- "GroundTruthImage.png" - the projected ground truth image in 2d from 3d point clouds
- "ground_truth_training_iterations.gif" - gif visualization of 3d gaussian splatting to obtain ground truth image
   "diffusion_training_iterations.gif" - gif visualization of optimization with 2d diffusion model

*Note: The code require CUDA (~10GB). Also you may have to increase optimization steps till 100000 to match results of official code. Takes around 15-20 to get output.*
