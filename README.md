# GaussianDreamer Demonstration (CSE252D)

-- By Bhavik Chandna and Isheta Bansal.

This repository presents a **CSE252D course demonstration** of **GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models**, originally developed by the [HuSVL group](https://github.com/hustvl/GaussianDreamer). It showcases how language prompts can be translated into 3D Gaussian splats using diffusion-based techniques.

📌 **Paper**: [CVPR 2024 Paper Link](https://arxiv.org/abs/2312.00768)  
📌 **Original Repository**: [hustvl/GaussianDreamer](https://github.com/hustvl/GaussianDreamer)

---

The actual repository of the paper has been fully integrated into [threestudio project](https://github.com/threestudio-project/threestudio) and has no clean code to understand each stage of the pipeline. This is the motivation of this repo. We have remove the threestudio dependancies and make the code simple and readable and integrated with gsplat. If you dont want to understand code and want to run a simple colab demo, you can head over to [Colab Link (Official Repo)](https://colab.research.google.com/drive/1PHdsF0PtqGb04FbcwAs_QIG_rNW7W4bh?usp=sharing)

## Method Summary

GaussianDreamer embarks on the task of **text-to-3D generation** using a two-stage pipeline:

1. **2D Diffusion Model**: Generates reference views of a 3D scene from a text prompt using a pre-trained image generator like Stable Diffusion.
2. **3D Gaussian Optimization**: Aligns a 3D Gaussian field to match the 2D views using differentiable splatting and novel regularization strategies.

This approach excels in producing coherent, colorful, and structurally plausible 3D assets from simple text inputs.

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
python main.py prompt sample_proportion
```
prompt = prompt as a string 
sample_proportion = value between 0.0 and 1.0 designating how many points to sample from the triangle mesh

Example: !python main.py "a shark" 0.5

All output can be found on the right under the same folder.

There are 5 output files:

"PointClouds.png" - visualization of generated point clouds
"PointCloudsViews.png" - visualization of generated point clouds from different camera angles
"GroundTruthImage.png" - the projected ground truth image in 2d from 3d point clouds
"ground_truth_training_iterations.gif" - gif visualization of 3d gaussian splatting to obtain ground truth image
"diffusion_training_iterations.gif" - gif visualization of optimization with 2d diffusion model

### Note: The code require CUDA (~10GB). Also you may have to increase optimization steps till 20000, I have done it for 1000 to show decent results.
