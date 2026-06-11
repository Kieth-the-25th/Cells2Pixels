import torch
import torch.nn.functional as F

from models.siren import Siren
from utils.camera import PerspectiveCamera

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.rsh import rsh_functions


class RaySampler:
    def __init__(self, HW, **kwargs):
        self.H, self.W = HW

        self.mode = kwargs.get("mode", "stride")
        assert self.mode in ['stride', 'patch']

        self.stride = kwargs.get('stride', 1)
        self.patch_size = kwargs.get('patch_size', min(32, self.H, self.W))

        self.stride_offset_x, self.stride_offset_y = None, None
        self.patch_offset_x, self.patch_offset_y = None, None

    def sample_rays(self, x: torch.Tensor):
        """
        x: Tensor Either (B, H, W, ...) when sampling a subset of rays shot from the camera
        """
        # Sample a subset of rays from the camera
        B = x.shape[0]
        if self.mode == 'stride':
            if self.stride_offset_x is None:
                self.stride_offset_x = np.random.randint(low=0, high=self.stride, size=B)
                self.stride_offset_y = np.random.randint(low=0, high=self.stride, size=B)

            return torch.stack([
                x[i, self.stride_offset_x[i]::self.stride, self.stride_offset_y[i]::self.stride]
                for i in range(B)
            ], dim=0)

        elif self.mode == 'patch':
            if self.patch_offset_x is None:
                self.patch_offset_x = np.random.randint(low=0, high=self.H - self.patch_size + 1, size=B)
                self.patch_offset_y = np.random.randint(low=0, high=self.W - self.patch_size + 1, size=B)

            return torch.stack([
                x[i, self.patch_offset_x[i]:self.patch_offset_x[i] + self.patch_size,
                self.patch_offset_y[i]:self.patch_offset_y[i] + self.patch_size]
                for i in range(B)
            ], dim=0)

    def sample_pixels(self, x: torch.Tensor):
        """
        x: Tensor with shape (B, C, H, W) or
            with shape (B, num_views, C, H, W) when sampling a subset of pixels from the rendered image.
        """
        B = x.shape[0]
        if self.mode == 'stride':
            if self.stride_offset_x is None:
                self.stride_offset_x = np.random.randint(low=0, high=self.stride, size=B)
                self.stride_offset_y = np.random.randint(low=0, high=self.stride, size=B)

            return torch.stack([
                x[i, ..., self.stride_offset_x[i]::self.stride, self.stride_offset_y[i]::self.stride]
                for i in range(B)
            ], dim=0)

        elif self.mode == 'patch':
            if self.patch_offset_x is None:
                self.patch_offset_x = np.random.randint(low=0, high=self.H - self.patch_size + 1, size=B)
                self.patch_offset_y = np.random.randint(low=0, high=self.W - self.patch_size + 1, size=B)

            return torch.stack([
                x[i, ..., self.patch_offset_x[i]:self.patch_offset_x[i] + self.patch_size,
                self.patch_offset_y[i]:self.patch_offset_y[i] + self.patch_size]
                for i in range(B)
            ], dim=0)

def get_living_mask(s):
    K = 3  # kernel size
    alpha = s[..., 3:4]
    return torch.nn.functional.max_pool3d(alpha, K, stride=1, padding=K // 2) > 0.1

class RendererRF:
    def __init__(self, num_samples: int = 64, perturb: bool = False,
                 voxel_bounds: tuple = (-1.0, 1.0), background_color=1.0,
                 density_factor=4.0, color_factor=2.0,
                 apply_living_mask: bool = False,
                 max_rays_per_chunk: int = 16384,  # For large voxel grids, limit the number of rays processed at once
                 non_linearity: str = "softplus",  # The non-linearity applied to get a positive-value density
                 sampler_kwargs: dict = None,
                 num_fine_samples: int = 0,
                 sh_degree=0,
                 **kwargs,  # Just for backward compatibility, ignore
                 ):
        self.num_samples = num_samples
        self.perturb = perturb
        self.voxel_bounds = voxel_bounds
        self.background_color = background_color
        self.density_factor = density_factor
        self.color_factor = color_factor
        self.apply_living_mask = apply_living_mask
        self.max_rays_per_chunk = max_rays_per_chunk
        self.non_linearity = {
            "softplus": F.softplus,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
        }[non_linearity]
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs is not None else {}
        self.num_fine_samples = num_fine_samples
        self.sh_degree = sh_degree

        self.rsh_func = rsh_functions[sh_degree]

    @staticmethod
    def trilinear_sample_voxel(voxel_grid: torch.Tensor,
                               sample_points: torch.Tensor,
                               bounds: tuple = (-1.0, 1.0)):
        """Trilinearly interpolate voxel_grid at the query locations.

        Args:
            voxel_grid (B, C, H, W, D): Feature grid.
            sample_points (B, h, w, N, 3): Query positions in world coordinates.
            bounds: Tuple of (min, max) world coordinate defining the cube that contains the grid.

        Returns:
            features (B, h, w, N, C): Interpolated features.
            rel_coords (B, h, w, N, 3): Relative within voxel coordinates in [-1, 1].
        """
        B, C, H, W, D = voxel_grid.shape
        min_bound, max_bound = bounds

        # Map world coordinates to normalised [-1, 1] cube expected by grid_sample.
        sample_points = 2.0 * (sample_points - min_bound) / (max_bound - min_bound) - 1.0  # still (B, h, w, N, 3)

        if sample_points.shape[0] == 1:
            sample_points = sample_points.expand(B, -1, -1, -1, -1)  # Expand to match voxel_grid batch size

        # Perform interpolation.
        # https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/3
        sampled_features = F.grid_sample(voxel_grid.permute(0, 1, 4, 3, 2),  # (B, C, H, W, D) to (B, C, D, W, H)
                                         sample_points,
                                         mode="bilinear",
                                         align_corners=False)
        # sampled_features has shape (B, C, H, W, N)
        sampled_features = sampled_features.permute(0, 2, 3, 4, 1)  # (B, h, w, N, C)

        # Compute within cell relative coordinates in [0, 1]
        idx = (sample_points + 1.0) / 2.0  # Normalize to [0, 1]
        idx = idx * torch.tensor([W - 1, H - 1, D - 1], device=sample_points.device)
        relative_coords = (idx - idx.floor()) * 2.0 - 1.0  # Convert to [-1, 1] range

        return sampled_features, relative_coords

    # Hierarchical sampling from NeRF
    @staticmethod
    @torch.no_grad()
    def sample_pdf(bins, weights, N_samples, det=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def render(self,
               voxel_grid: torch.Tensor,
               camera: PerspectiveCamera,
               siren: Siren,
               batchify_rays: bool = False,
               num_samples: int = None,
               perturb: bool = None,
               voxel_bounds: tuple = None,
               background_color=None,
               density_factor=None,
               color_factor=None,
               apply_living_mask: bool = None,
               sampler_kwargs: dict = None,
               num_fine_samples: int = None,
               ):
        """Render an RGB image from *voxel_grid* using *camera* and *siren*.

        Parameters
        ----------
        voxel_grid : (B,C,H,W,D) tensor  batched feature grids
        camera     : PerspectiveCamera instance with n_view different cameras
        siren      : neural renderer f(V, Δx)→(rgb, σ)
        num_samples: int  points per ray
        perturb    : bool  stratified depth sampling like NeRF
        voxel_bounds: tuple of (min, max) world coordinates defining the cube that contains the grid
        background_color: float  0.0 for black, 1.0 for white
        density_factor: float  scaling factor for the occupancy density
        color_factor: float  scaling factor for the RGB output
        apply_living_mask: bool  if True, multiply the occupancy density by the living mask (features[..., 3:4])
        sampler_kwargs: dict  additional arguments for the ray sampling function

        Returns
        -------
        rgb   : (B,num_views,3,h,w)
        depth : (B,num_views,1,h,w)
        opacity: (B,num_views,1,h,w) This is not differentiable w.r.t the living mask.
        opacity_living: (B,num_views,h,w) if apply_living_mask is True. This is differentiable w.r.t the living mask.
        sampler: RaySampler instance used for sampling the rays (deterministic sampling given the instance).
        """
        B, C, H, W, D = voxel_grid.shape
        num_views = camera.position.shape[0]
        # voxel_grid = voxel_grid.repeat(num_views, 1, 1, 1, 1)  # (B*num_views, C, H, W, D)
        voxel_grid = torch.repeat_interleave(voxel_grid, num_views, dim=0)  # (B*num_views, C, H, W, D)

        num_samples = num_samples or self.num_samples
        perturb = perturb if perturb is not None else self.perturb
        voxel_bounds = voxel_bounds or self.voxel_bounds
        background_color = background_color if background_color is not None else self.background_color
        density_factor = density_factor if density_factor is not None else self.density_factor
        color_factor = color_factor if color_factor is not None else self.color_factor
        apply_living_mask = apply_living_mask if apply_living_mask is not None else self.apply_living_mask
        sampler_kwargs = sampler_kwargs if sampler_kwargs is not None else self.sampler_kwargs
        num_fine_samples = num_fine_samples if num_fine_samples is not None else self.num_fine_samples

        def render_ray_chunk(spc, sdc, roc, rdc):
            """
            spc: sample points chunk (B*num_views, h, w, N, 3)
            sdc: sample depths chunk (B*num_views, h, w, N, 1)
            roc: ray origins chunk (B*num_views, h, w, 3)
            rdc: ray directions chunk (B*num_views, h, w, 3)
            """
            _, h, w, N, _ = spc.shape  # (B*num_views, h, w, N, 3)
            low, high = voxel_bounds

            sh_evals = self.rsh_func(rdc)  # (B*num_views, h, w, (sh_degree+1)**2)
            sh_evals = sh_evals.unsqueeze(-2) # (B*num_views, h, w, 1, (sh_degree+1)**2)
            sh_evals = sh_evals.repeat_interleave(3, -1) # (B*num_views, h, w, 1, 3*(sh_degree+1)**2)

            deltas = sdc[..., 1:, :] - sdc[..., :-1, :]
            last_delta = 1e10 * torch.ones_like(deltas[..., :1, :])
            deltas = torch.cat([deltas, last_delta], dim=-2)  # (B*num_views, h, w, N, 1)

            # 2. voxel coarse lookup
            if num_fine_samples > 0 and apply_living_mask:
                inside = ((spc >= low) & (spc <= high)).all(dim=-1).unsqueeze(-1)  # (B*num_views, h, w, N, 1)
                living_mask, relative_coords = self.trilinear_sample_voxel(voxel_grid[:, 3:4], spc, voxel_bounds)
                living_mask = torch.relu(living_mask)
                living_mask = living_mask * inside.float()  # Mask out points outside the voxel bounds
                alpha_living = 1.0 - torch.exp(-living_mask * deltas)
                transmittance_living = torch.cumprod(torch.cat(
                    [torch.ones_like(alpha_living[..., :1, :]), 1. - alpha_living + 1e-10], dim=-2),
                    dim=-2)[..., :-1, :]
                weights_living = (alpha_living * transmittance_living)  # (B*num_views, h, w, N, 1)

                sdc_midpoint = 0.5 * (sdc[..., 1:, 0] + sdc[..., :-1, 0])

                sdc_midpoint = sdc_midpoint.reshape(-1, N - 1)
                weights_living = weights_living.squeeze(-1)[..., 1:-1].reshape(-1, N - 2)
                fine_samples_depth = self.sample_pdf(sdc_midpoint, weights_living,
                                                     num_fine_samples, det=not perturb)  # [B*num_views*h*w, N_fine]
                fine_samples_depth = fine_samples_depth.reshape(-1, h, w, num_fine_samples)
                fine_samples_depth = fine_samples_depth.unsqueeze(-1)  # (B*num_views, h, w, N_fine, 1)

                N = N + num_fine_samples

                sdc, _ = torch.sort(torch.cat([sdc, fine_samples_depth], dim=-2), dim=-2)
                spc = sdc * rdc[..., None, :] + roc[..., None, :]

                deltas = sdc[..., 1:, :] - sdc[..., :-1, :]
                last_delta = 1e10 * torch.ones_like(deltas[..., :1, :])
                deltas = torch.cat([deltas, last_delta], dim=-2)  # (B*num_views, h, w, N, 1)



            living_mask_voxel = (torch.nn.functional.max_pool3d(voxel_grid[:, 3:4], 3, stride=1, padding=1) > 0.1).float()
            voxel_aug = torch.cat([living_mask_voxel, voxel_grid], dim=1)

            features, relative_coords = self.trilinear_sample_voxel(voxel_aug, spc, voxel_bounds)
            features, binary_living_mask = features[..., 1:], features[..., :1]
            # features: (B*num_views, h, w, N, C)
            inside = ((spc >= low) & (spc <= high)).all(dim=-1).unsqueeze(-1)  # (B*num_views, h, w, N, 1)

            # 3. neural renderer
            with torch.autocast(device_type=features.device.type, dtype=torch.float16):
                raw = siren(features, relative_coords)  # (B*num_views, h, w, N, 4)

            sh_coeffs = raw[..., :-1] # (B*num_views, h, w, N, 3*(sh_degree+1)**2)
            sh_colors = sh_coeffs * color_factor * sh_evals
            r = sh_colors[..., 0::3].sum(dim=-1, keepdim=True)
            g = sh_colors[..., 1::3].sum(dim=-1, keepdim=True)
            b = sh_colors[..., 2::3].sum(dim=-1, keepdim=True)
            rgb = torch.cat([r, g, b], dim=-1) # (B*num_views, h, w, N, 3)
            rgb = torch.relu(rgb)
            # rgb = torch.sigmoid(raw[..., :3] * color_factor)  # (B*num_views, h, w, N, 3)
            sigma = self.non_linearity(raw[..., 3:4]) * density_factor  # (B*num_views, h, w, N, 1) occupancy density
            #     sigma = F.softplus(raw[..., 3:4]) * 1.0  # (B*num_views, h, w, N, 1) occupancy density
            sigma = sigma * inside.float()  # Mask out points outside the voxel bounds
            if apply_living_mask:
                sigma = sigma * (binary_living_mask > 0.1).float()
                living_mask = torch.relu(features[..., 3:4])
                living_mask = living_mask * inside.float()  # Mask out points outside the voxel bounds

            # 4. volumetric integration (hierarchical alpha compositing)

            alpha = 1.0 - torch.exp(-sigma * deltas)
            transmittance = torch.cumprod(torch.cat(
                [torch.ones_like(alpha[..., :1, :]), 1. - alpha + 1e-10], dim=-2),
                dim=-2)[..., :-1, :]
            weights = (alpha * transmittance)  # (B,H,W,N,1)

            rgb_map = (weights * rgb).sum(dim=-2)  # (B*num_views,H,W,3)
            depth_map = (weights * sdc).sum(dim=-2)  # (B*num_views,H,W,1)
            acc_map = weights.sum(dim=-2)  # (B*num_views,H,W,1)

            rgb_final = rgb_map + (1. - acc_map) * background_color

            # print(depth_map.shape, weights.shape)
            # depth_map = 1.0 / torch.max(1e-6 * torch.ones_like(depth_map), depth_map / weights.sum(dim=-2).clamp(1e-6))

            rgb_final = rgb_final.view(B, num_views, h, w, 3).permute(0, 1, 4, 2, 3)  # (B, num_views, 3, h, w)
            depth_map = depth_map.view(B, num_views, h, w, 1).permute(0, 1, 4, 2, 3)  # (B, num_views, 1, h, w)
            acc_map = acc_map.view(B, num_views, h, w, 1).permute(0, 1, 4, 2, 3)  # (B, num_views, 1, h, w)

            acc_map_living = None
            if apply_living_mask:
                alpha_living = 1.0 - torch.exp(-living_mask * deltas)
                transmittance_living = torch.cumprod(torch.cat(
                    [torch.ones_like(alpha_living[..., :1, :]), 1. - alpha_living + 1e-10], dim=-2),
                    dim=-2)[..., :-1, :]
                weights_living = (alpha_living * transmittance_living)  # (B*num_views, h, w, N, 1)
                acc_map_living = weights_living.sum(dim=-2)  # (B*num_views,H,W,1)

                # (B, num_views, 1, h, w)
                acc_map_living = acc_map_living.view(B, num_views, h, w, 1).permute(0, 1, 4, 2, 3)

            return rgb_final, depth_map, acc_map, acc_map_living

        # 1. sample points along rays

        (sample_points, sample_depths,
         ray_origins, ray_dirs) = camera.sample_along_rays(num_samples=num_samples, perturb=perturb, voxel_bounds=voxel_bounds)
        
        sampler = RaySampler(sample_points.shape[1:3], **sampler_kwargs)
        sample_points = sampler.sample_rays(sample_points)  # (num_views, h, w, N, 3)
        sample_depths = sampler.sample_rays(sample_depths)  # (num_views, h, w, N, 1)
        sample_points = sample_points.repeat(B, 1, 1, 1, 1)  # (B*num_views, h, w, N, 3)
        sample_depths = sample_depths.repeat(B, 1, 1, 1, 1)  # (B*num_views, h, w, N, 1)
        ray_origins = sampler.sample_rays(ray_origins)  # (num_views, h, w, 3)
        ray_dirs = sampler.sample_rays(ray_dirs)  # (num_views, h, w, 3)
        ray_origins = ray_origins.repeat(B, 1, 1, 1)  # (B*num_views, h, w, 3)
        ray_dirs = ray_dirs.repeat(B, 1, 1, 1)  # (B*num_views, h, w, 3)

        if not batchify_rays:
            return *render_ray_chunk(sample_points, sample_depths, ray_origins, ray_dirs), sampler

        _, h, w, N, _ = sample_points.shape
        sample_points = sample_points.reshape(B * num_views, h * w, 1, N, 3)  # (B*num_views, h*w, 1, N, 3)
        sample_depths = sample_depths.reshape(B * num_views, h * w, 1, N, 1)  # (B*num_views, h*w, 1, N, 1)
        ray_origins = ray_origins.reshape(B * num_views, h * w, 1, 3)
        ray_dirs = ray_dirs.reshape(B * num_views, h * w, 1, 3)

        rgb_chunks, depth_chunks, acc_chunks, acc_living_chunks = [], [], [], []
        for i in range(0, h * w, self.max_rays_per_chunk):
            end = min(i + self.max_rays_per_chunk, h * w)
            spc = sample_points[:, i:end, :, :, :]
            sdc = sample_depths[:, i:end, :, :, :]
            roc = ray_origins[:, i:end, :, :]
            rdc = ray_dirs[:, i:end, :, :]
            rgb_chunk, depth_chunk, acc_chunk, acc_living_chunk = render_ray_chunk(spc, sdc, roc, rdc)
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
            acc_chunks.append(acc_chunk)
            acc_living_chunks.append(acc_living_chunk)

        rgb = torch.cat(rgb_chunks, dim=3).view(B, num_views, 3, h, w)
        depth = torch.cat(depth_chunks, dim=3).view(B, num_views, 1, h, w)
        opacity = torch.cat(acc_chunks, dim=3).view(B, num_views, 1, h, w)
        if apply_living_mask:
            opacity_living = torch.cat(acc_living_chunks, dim=3).view(B, num_views, 1, h, w)
        else:
            opacity_living = None

        return rgb, depth, opacity, opacity_living, sampler

    @staticmethod
    @torch.no_grad()
    def to_pil(rendered_features, batch_stack='vertical', view_stack='vertical', target_stack='horizontal'):
        """
        :param rendered_features: Tuple of (rgb, depth, occupancy, opacity_living)
        rgb: (B, num_views, 3, h, w)
        depth: (B, num_views, 1, h, w) or None
        opacity: (B, num_views, 1, h, w) or None
        opacity_living: (B, num_views, h, w) or None

        :param batch_stack: Whether to stack the batch elements vertically or horizontally.
        :param view_stack: Whether to stack the views vertically or horizontally.
        :param target_stack: Whether to stack the target channels vertically or horizontally.

        :return: A PIL Image showing the rendered images.
        """
        stack_batch = np.vstack if batch_stack == 'vertical' else np.hstack
        stack_view = np.vstack if view_stack == 'vertical' else np.hstack
        stack_target = np.vstack if target_stack == 'vertical' else np.hstack

        if len(rendered_features) == 1:
            rgb, depth, opacity, opacity_living = rendered_features[0], None, None, None
        else:
            rgb, depth, opacity, opacity_living = rendered_features

        assert rgb.dim() == 5, "The input tensor should have 5 dimensions"
        batch_size, num_views, _, height, width = rgb.shape

        image_list = []
        for x in [rgb, depth, opacity, opacity_living]:
            if x is None:
                continue

            x = x.clip(0.0, 1.0)  # Ensure values are in [0, 1]

            if x.shape[2] == 1:
                x = x.expand(-1, -1, 3, -1, -1)
            x = (x.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)
            image_list.append(
                stack_batch(
                    [stack_view(x[i]) for i in range(batch_size)]
                )  # [batch_size*height, num_views*width, num_features]
            )

        image = Image.fromarray(stack_target(image_list))
        return image


if __name__ == "__main__":
    from utils.misc import auto_device
    device = auto_device()
    with torch.no_grad():
        chn = 12
        
        siren = Siren(chn, coord_dim=3, out_features=4, fx="linear",
                      activation="softplus", outermost_linear=True).to(device)
        # siren = lambda x, y: x[..., :4]  # Mock siren for testing
        camera = PerspectiveCamera(
            fov=60.0, elevation=[0.0, 35.0], azimuth=[0.0, 45.0], distance=[4.0, 4.0], bounds=[1.0, 8.0],
            height=256, width=256, device=device,
        )
        obj = "hotdog"
        camera, keys = PerspectiveCamera.from_json(f"data/radiance_fields/{obj}/train_camera_params.json",
                                                   device=device, return_keys=True)
        camera.height, camera.width = 64, 64

        view_idx = [0, 1][:1]
        camera = camera.sample_batch(len(view_idx), view_idx)  # Sample a single camera from the batch

        batch_size = 1
        x = torch.rand(batch_size, chn, 64, 64, 64, device=device)  # Example voxel grid (B, C, H, W, D)
        x = x * 0.0
        r = 5
        x[:, :, 32 - r:32 + r + 1, 32 - r:32 + r + 1, 32 - r:32 + r + 1] = 1.0
        # x = torch.zeros(1, chn, 16, 16, 16, device=device)  # Example voxel grid (B, C, H, W, D)
        # for i in range(16):
        #     x[:, :, :, i, :] = i / 16.0  # Fill with some pattern for testing
        stride = 1
        print(camera.distance[0])
        renderer = RendererRF(num_samples=128, num_fine_samples=128, perturb=False, voxel_bounds=(-1.0, 1.0),
                              background_color=0.0,
                              density_factor=1.0, color_factor=2.0, apply_living_mask=True,
                              sampler_kwargs={"mode": "stride", "stride": stride})
        rgb, depth, opacity, opacity_living, sampler = renderer.render(x, camera, siren, batchify_rays=True,
                                                                       sampler_kwargs={"mode": "stride",
                                                                                       "stride": stride})

        rgb = torch.clamp(rgb, 0.0, 1.0) * 1.0  # Ensure RGB values are in [0, 1]

        depth = torch.clamp(depth / 3.0, 0.0, 1.0)

        stride = 2
        target_imgs = [
            np.array(Image.open(f"data/radiance_fields/{obj}/train/{keys[idx]}").convert("RGB").resize(
                (camera.height, camera.width)))
            for idx in view_idx]
        target_imgs = np.stack(target_imgs, axis=0)  # (num_views, h, w, C)
        target_imgs = torch.tensor(target_imgs, dtype=torch.float32, device=device) / 255.0
        target_imgs = target_imgs.permute(0, 3, 1, 2).unsqueeze(0)  # (1, num_views, h, w, C)
        target_imgs = sampler.sample_pixels(target_imgs)  # Sample the target images using the same sampler

        renderer.to_pil((0.5 * (rgb + target_imgs), None, opacity, opacity_living)).show()
