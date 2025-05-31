import torch
import math
import numpy as np
import os
from gsplat import project_gaussians, rasterize_gaussians
from torch import optim
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from torch.nn.functional import mse_loss

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[0], q[1], q[2], q[3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    R = np.array([
        [ww + xx - yy - zz,      2 * (xy - zw),      2 * (xz + yw)],
        [    2 * (xy + zw),  ww - xx + yy - zz,      2 * (yz - xw)],
        [    2 * (xz - yw),      2 * (yz + xw),  ww - xx - yy + zz]
    ], dtype=np.float32)

    return R


class OptimizationWith2DModel:

    def __init__(self, prompt, positions, colors, scales, rotations, opacities):
        self.prompt = prompt
        self.initialize_parameters(positions, colors, scales, rotations, opacities)
        self.prepare_2d_diffusion_model()
        self.get_ground_truth_image()

    def initialize_parameters(self, positions, colors, scales, rotations, opacities):
ss        self.positions = torch.tensor(positions, dtype=torch.float32, device='cuda')
        self.colors    = torch.tensor(colors, dtype=torch.float32, device='cuda')
        self.scales    = torch.tensor(scales, dtype=torch.float32, device='cuda')
        self.quats     = torch.tensor(rotations, dtype=torch.float32, device='cuda')
        self.opacities = torch.tensor(opacities, dtype=torch.float32, device='cuda')

        self.positions.requires_grad  = True
        self.colors.requires_grad     = True
        self.scales.requires_grad     = True
        self.quats.requires_grad      = True
        self.opacities.requires_grad  = True

        self.glob_scale = float(1.0)
        self.viewmat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device="cuda")
        self.viewmat.requires_grad = False

        self.fx = 0.5 * 512.0 / math.tan(0.5 * (math.pi / 2.0))
        self.fy = 0.5 * 512.0 / math.tan(0.5 * (math.pi / 2.0))
        self.cx = float(256.0)
        self.cy = float(256.0)
        self.img_height = 512
        self.img_width  = 512
        self.block_width = 16

        self.scheduler = None
        self.pipe      = None
        self.autoencoder = None
        self.denoiser    = None

        self.prompt_embeddings = self.get_prompt_text_embeddings()

    def get_ground_truth_image(self):
        # ... (same as before) ...
        with torch.no_grad():
            eye_fov = 60.0 * math.pi / 180.0
            zNear = -0.1
            zFar  = -50

            t_val = abs(zNear) * math.tan(eye_fov / 2)
            r_val = t_val
            l_val = -r_val
            b_val = -t_val
            perspmat = torch.tensor([
                [2 * zNear / (r_val - l_val), 0.0, -(r_val + l_val) / (r_val - l_val), 0.0],
                [0.0, 2 * zNear / (t_val - b_val), -(t_val + b_val) / (t_val - b_val), 0.0],
                [0.0, 0.0, (zNear + zFar) / (zNear - zFar), -2 * zNear * zFar / (zNear - zFar)],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=torch.float32, device="cuda")

            viewportmat = torch.tensor([
                [256.0, 0.0, 0.0, 256.0],
                [0.0, 256.0, 0.0, 256.0],
                [0.0, 0.0, 1.0,   0.0],
                [0.0, 0.0, 0.0,   1.0]
            ], dtype=torch.float32, device="cuda")

            viewmat = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32, device="cuda")
            perspective_projection_matrix = torch.matmul(perspmat, viewmat)

            ones = torch.ones((self.positions.shape[0], 1), device="cuda")
            positions_h = torch.cat((self.positions, ones), dim=1)

            def apply_mat(pos):
                return torch.matmul(perspective_projection_matrix, pos)

            perspective_positions = torch.stack([apply_mat(p) for p in positions_h])[:, :3]

            perspective_positions -= perspective_positions.min()
            perspective_positions /= perspective_positions.max()
            perspective_positions = 2 * perspective_positions - 1
            perspective_positions_h = torch.cat((perspective_positions, ones), dim=1)

            def apply_view(pos):
                return torch.matmul(viewportmat, pos)

            ground_truth_positions = torch.stack([apply_view(p) for p in perspective_positions_h])[:, :3]

            self.ground_truth_image = torch.ones((3, 512, 512), device="cuda")
            z_buffer = torch.full((512, 512), math.inf, device="cuda")

            N = self.positions.shape[0]
            for i in range(N):
                x, y, z = ground_truth_positions[i]
                x = math.floor(x.item()) - 1
                y = 511 - math.floor(y.item())

                if 0 <= x < 512 and 0 <= y < 512:
                    if z <= z_buffer[x, y]:
                        self.ground_truth_image[0, x, y] = self.colors[i, 0]
                        self.ground_truth_image[1, x, y] = self.colors[i, 1]
                        self.ground_truth_image[2, x, y] = self.colors[i, 2]
                        z_buffer[x, y] = z

            transform = transforms.ToPILImage()
            img = transform(self.ground_truth_image)

            os.makedirs("./outputs", exist_ok=True)
            img.save("./outputs/GroundTruthImage.png")

    def prepare_2d_diffusion_model(self):
        # ... (same as before) ...
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            subfolder="scheduler",
            torch_dtype=torch.float32
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            torch_dtype=torch.float32
        ).to("cuda")

        self.autoencoder = self.pipe.vae.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        self.denoiser = self.pipe.unet.eval()
        for param in self.denoiser.parameters():
            param.requires_grad_(False)

        self.weights = self.scheduler.alphas_cumprod.to("cuda")

    def train_ground_truth(self):
        optimizer = optim.Adam([
            {'params': [self.positions], 'lr': 0.00005},
            {'params': [self.colors],    'lr': 0.0125},
            {'params': [self.scales],    'lr': 0.001},
            {'params': [self.quats],     'lr': 0.01},
            {'params': [self.opacities], 'lr': 0.01}
        ])

        self.ground_truth_training_images = []
        for it in range(6000):
            optimizer.zero_grad()
            gauss_2d, depths, radii, conics, compensation, num_tiles, cov_3d = project_gaussians(
                self.positions,
                self.scales,
                self.glob_scale,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.viewmat,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.img_height,
                self.img_width,
                self.block_width
            )

            rendered = rasterize_gaussians(
                gauss_2d,
                depths,
                radii,
                conics,
                num_tiles,
                self.colors,
                self.opacities,
                self.img_height,
                self.img_width,
                self.block_width
            )
            rendered = rendered.permute(2, 0, 1)

            loss = mse_loss(rendered, self.ground_truth_image)
            loss.backward()
            optimizer.step()

            if it % 100 == 0:
                transform = transforms.ToPILImage()
                img = transform(rendered)
                self.ground_truth_training_images.append(img)
                print(f"Iteration: {it}/6000   Loss: {loss.item():.6f}")

        os.makedirs("./outputs", exist_ok=True)
        self.ground_truth_training_images[0].save(
            "./outputs/ground_truth_training_iterations.gif",
            save_all=True,
            append_images=self.ground_truth_training_images[1:],
            optimize=False,
            duration=10,
            loop=0
        )

    def train(self):
        optimizer = optim.Adam([
            {'params': [self.positions], 'lr': 0.00005},
            {'params': [self.colors],    'lr': 0.0125},
            {'params': [self.scales],    'lr': 0.001},
            {'params': [self.quats],     'lr': 0.01},
            {'params': [self.opacities], 'lr': 0.01}
        ])

        self.sds_training_images = []
        for it in range(1000):
            optimizer.zero_grad()
            gauss_2d, depths, radii, conics, compensation, num_tiles, cov_3d = project_gaussians(
                self.positions,
                self.scales,
                self.glob_scale,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.viewmat,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.img_height,
                self.img_width,
                self.block_width
            )

            rendered = rasterize_gaussians(
                gauss_2d,
                depths,
                radii,
                conics,
                num_tiles,
                self.colors,
                self.opacities,
                self.img_height,
                self.img_width,
                self.block_width
            )
            rendered = rendered.permute(2, 0, 1).unsqueeze(0)

            latents = self.image_to_latents(rendered).to("cuda")
            t = torch.randint(20, 981, [1], dtype=torch.long, device="cuda")

            sds_grad = self.calculate_sds_gradient(latents, t)
            sds_grad = torch.nan_to_num(sds_grad)

            sds_loss = mse_loss(latents, latents - sds_grad, reduction='sum')
            sds_loss.backward()
            optimizer.step()

            if it % 50 == 0:
                transform = transforms.ToPILImage()
                img = transform(rendered.squeeze())
                self.sds_training_images.append(img)
                print(f"Iteration: {it}/1000   Loss: {sds_loss.item():.6f}")

        os.makedirs("./outputs", exist_ok=True)
        self.sds_training_images[0].save(
            "./outputs/diffusion_training_iterations.gif",
            save_all=True,
            append_images=self.sds_training_images[1:],
            optimize=False,
            duration=10,
            loop=0
        )

    def calculate_sds_gradient(self, latents, t):
        with torch.no_grad():
            actual_noise = torch.randn(latents.size(), device="cuda")
            noisy_latents = self.scheduler.add_noise(latents, actual_noise, t)
            predicted_noise = self.denoiser_forward(noisy_latents, t, self.prompt_embeddings)

        sds_grad = (1 - self.weights[t]).view(-1, 1, 1, 1) * (predicted_noise - actual_noise)
        return sds_grad

    def image_to_latents(self, image: torch.Tensor) -> torch.Tensor:
        scaled = image * 2.0 - 1.0
        encoded = self.autoencoder.encode(scaled.to(torch.float32)).latent_dist
        latents = (encoded.sample() * 0.18215).to(torch.float32)
        return latents

    def get_prompt_text_embeddings(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="text_encoder"
        ).to("cuda")
        for param in text_encoder.parameters():
            param.requires_grad_(False)

        tokens = tokenizer([self.prompt], padding="max_length", max_length=77, return_tensors="pt")
        with torch.no_grad():
            embeddings = text_encoder(tokens.input_ids.to("cuda"))[0]
        return embeddings

    def denoiser_forward(self, latents, t, text_embeddings):
        return self.denoiser(
            latents.to(torch.float32),
            t.to(torch.float32),
            encoder_hidden_states=text_embeddings.to(torch.float32)
        ).sample.to(torch.float32)

    # def save_gaussians_as_splat(self, filename: str = "./outputs/scene.splat"):
    #     """
    #     Write all Gaussian parameters into a single .splat file.
    #     Format:
    #       [int32 N]  (number of Gaussians)
    #       Then for each Gaussian, 14 floats (all float32):
    #         [pos_x, pos_y, pos_z,
    #          scale_x, scale_y, scale_z,
    #          quat_x, quat_y, quat_z, quat_w,
    #          color_r, color_g, color_b,
    #          opacity]
    #     """
    #     positions = self.positions.detach().cpu().numpy()
    #     scales    = self.scales.detach().cpu().numpy()
    #     quats     = self.quats.detach().cpu().numpy()
    #     colors    = self.colors.detach().cpu().numpy()
    #     opacities = self.opacities.detach().cpu().numpy()
    #     N = positions.shape[0]
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     with open(filename, "wb") as f:
    #         # write number of Gaussians as int32
    #         f.write(np.int32(N).tobytes())
    #         for i in range(N):
    #             arr = np.concatenate([
    #                 positions[i],        # 3 floats
    #                 scales[i],           # 3 floats
    #                 quats[i],            # 4 floats
    #                 colors[i],           # 3 floats
    #                 [opacities[i].item()]       # 1 float
    #             ]).astype(np.float32)
    #             f.write(arr.tobytes())

    #     print(f"Saved {N} Gaussians to {filename}")

    # def export_splat(self, filename: str = "./outputs/scene.splat"):
    #     self.save_gaussians_as_splat(filename)
