import sys

from src.DiffusionModel3D import DiffusionModel3D
from src.GaussianInitialization import GaussianInitialization
from src.OptimizationWith2DModel import OptimizationWith2DModel

if __name__ == "__main__":
    prompt = sys.argv[1]
    sample_proportion = float(sys.argv[2])

    diffusion_model_3d = DiffusionModel3D(prompt)
    diffusion_model_3d.generateTriangleMesh()
    triangle_mesh = diffusion_model_3d.mesh

    print("Triangle mesh generated using 3d diffusion model")
    print("# of triangles in mesh: {0}".format(triangle_mesh.vertices.shape[0]))

    diffusion_model_3d.generatePointClouds(sample_proportion)
    point_clouds = diffusion_model_3d.point_clouds

    print("Point clouds generated with random weighted probabilities")
    print("# of point clouds: {0}".format(point_clouds.vertices.shape[0]))

    diffusion_model_3d.visualizePointClouds()
    print("Point clouds visualization saved as \"PointClouds.png\"")
    diffusion_model_3d.visualizePointCloudsViews()
    print("Point clouds (from different camera angles) visualization saved as \"PointCloudsViews.png\"")

    gaussian_initialization = GaussianInitialization(point_clouds.vertices, point_clouds.colors)
    gaussians = gaussian_initialization.initializeGaussians()

    print("3D Gaussians initialized")

    diffusion_model_2d = OptimizationWith2DModel("a shark",
                                                 gaussians.positions,
                                                 gaussians.colors,
                                                 gaussians.scales,
                                                 gaussians.rotations,
                                                 gaussians.alphas)
    print("Applying perspective projection")
    print("Obtaining ground truth image through 3D gaussian splatting")
    diffusion_model_2d.train_ground_truth()
    print("Ground truth training iteration images saved as \"ground_truth_training_iterations.gif\"")

    print("Optimizing with 2d diffusion model")
    diffusion_model_2d.train()
    print("Diffusion model training iteration images saved as \"diffusion_training_iterations.gif\"")




