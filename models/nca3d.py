import torch


def depthwise_conv(x, filters, padding="circular"):
    """filters: [filter_n, h, w, d]"""
    b, ch, h, w, d = x.shape
    y = x.reshape(b * ch, 1, h, w, d)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1, 1, 1], padding)
    y = torch.nn.functional.conv3d(y, filters[:, None])
    return y.reshape(b, -1, h, w, d)


class VNCA(torch.nn.Module):
    def __init__(
        self,
        channels=12,
        fc_dim=128,
        noise_level=0.0,
        update_prob=0.5,
        device=None,
        precision=torch.float32,
        padding="circular",
    ):
        super().__init__()
        self.channels = channels
        self.update_prob = update_prob
        self.device = device
        self.perception_kernels = 5
        self.precision = precision
        self.padding = padding
        self.register_buffer(
            "noise_level", torch.tensor([noise_level], dtype=self.precision)
        )

        input_dim = (self.channels + 0) * 5
        self.w1 = torch.nn.Conv3d(input_dim, fc_dim, 1, bias=True, device=self.device)
        self.w2 = torch.nn.Conv3d(
            fc_dim, self.channels, 1, bias=False, device=self.device
        )

        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
        torch.nn.init.zeros_(self.w2.weight)

        with torch.no_grad():
            delta_one = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                device=self.device,
                dtype=self.precision,
            )
            delta_two = torch.tensor(
                [[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0], [-2.0, 0.0, 2.0]],
                device=self.device,
                dtype=self.precision,
            )
            sobel_z = torch.stack([delta_one, delta_two, delta_one]) / 2.0
            sobel_y = sobel_z.permute(0, 2, 1)
            sobel_x = sobel_z.permute(2, 1, 0)

            lap1 = torch.tensor(
                [[2.0, 3.0, 2.0], [3.0, 6.0, 3.0], [2.0, 3.0, 2.0]],
                device=self.device,
                dtype=self.precision,
            )
            lap2 = torch.tensor(
                [[3.0, 6.0, 3.0], [6.0, -88.0, 6.0], [3.0, 6.0, 3.0]],
                device=self.device,
                dtype=self.precision,
            )
            lap = torch.stack([lap1, lap2, lap1])
            lap = lap / 8.0

            ident = torch.zeros(3, 3, 3, device=self.device, dtype=self.precision)
            ident[1, 1, 1] = 1.0

            self.filters = torch.stack([ident, sobel_x, sobel_y, sobel_z, lap])

    def forward(self, s):
        b, c, h, w, d = s.shape
        z = depthwise_conv(s, self.filters, padding=self.padding)  # [b, 5 * chn, h, w]
        delta_s = self.w2(torch.relu(self.w1(z)))
        if self.update_prob < 1.0:
            update_mask = (
                torch.rand(b, 1, h, w, d, device=s.device, dtype=self.precision)
                < self.update_prob
            ).to(self.precision)
        else:
            update_mask = 1.0
        return s + delta_s * update_mask, z

    def seed(self, n, h=128, w=128, d=128):
        with torch.no_grad():
            return (
                torch.rand(
                    n, self.channels, h, w, d, device=self.device, dtype=self.precision
                )
                - 0.5
            ) * self.noise_level


class GrowingVNCA(VNCA):
    def __init__(self, seed_radius=3, **kwargs):
        super().__init__(**kwargs)
        self.seed_radius = seed_radius
        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.1)
        torch.nn.init.xavier_normal_(self.w2.weight, gain=0.1)

    def forward(self, s):
        """
        Computes one step of the NCA update rule using the specified integrator.
        The alive mask is updated based on the alpha channel.
        """
        pre_life_mask = GrowingVNCA.get_living_mask(s)
        new_s, z = super().forward(s)
        post_life_mask = GrowingVNCA.get_living_mask(new_s)

        new_s = new_s * torch.logical_and(pre_life_mask, post_life_mask).to(
            self.precision
        )
        return new_s, z

    @staticmethod
    def get_living_mask(s):
        K = 3  # kernel size
        alpha = s[:, 3:4]
        return torch.nn.functional.max_pool3d(alpha, K, stride=1, padding=K // 2) > 0.1

    def seed(self, n, h=128, w=128, d=128):
        s = torch.zeros(
            n, self.channels, h, w, d, device=self.device, dtype=self.precision
        )
        r = self.seed_radius - 1
        s[
            :,
            3:,
            h // 2 - r : h // 2 + r + 1,
            w // 2 - r : w // 2 + r + 1,
            d // 2 - r : d // 2 + r + 1,
        ] = 1.0

        return s


if __name__ == "__main__":
    from tqdm import tqdm
    from utils.camera import PerspectiveCamera
    from utils.volumetric_render import RendererRF
    from models.siren import Siren
    from utils.video import VideoWriter
    import numpy as np
    from PIL import Image
    from utils.misc import auto_device

    with VideoWriter() as video, torch.no_grad():
        chn = 8
        device = auto_device()
        siren = Siren(
            chn,
            coord_dim=3,
            out_features=4,
            fx="linear",
            activation="softplus",
            outermost_linear=True,
        ).to(device)
        camera = PerspectiveCamera(
            fov=60.0,
            elevation=[20.0],
            azimuth=[0.0],
            distance=[3.0],
            bounds=[1.0, 8.0],
            height=256,
            width=256,
            device=device,
        )

        camera = PerspectiveCamera.from_json(
            "data/radiance_fields/chair/train_camera_params.json", device=device
        )
        camera = camera.sample_batch(1)
        camera.height, camera.width = 256, 256

        model = GrowingVNCA(
            channels=chn,
            fc_dim=32,
            noise_level=0.0,
            seed_radius=3,
            update_prob=0.5,
            device=device,
        ).to(device)
        s = model.seed(1, h=64, w=64, d=64).to(device)  # Initial state

        renderer = RendererRF(
            num_samples=64,
            perturb=False,
            voxel_bounds=(-1.0, 1.0),
            apply_living_mask=True,
        )
        for _ in tqdm(range(128)):
            rgb, depth, _, opacity, _ = renderer.render(s, camera, siren)

            #rgb: [B, num_view, 3, H, W], depth: [B, num_view, 1, H, W], opacity: [B, num_view, 1, H, W]

            rgb = torch.clamp((rgb + 1.0) / 2.0, 0.0, 1.0)

            # rgb: [B, V, 3, H, W] -> take first batch and format as [V, H, W, 3]
            rgb_image = rgb[0].permute(0, 2, 3, 1).cpu().numpy()
            rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = np.hstack(rgb_image)

            # opacity: [B, V, 1, H, W] -> take first batch and format as [V, H, W, 3]
            opacity_image = opacity[0].permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
            opacity_image = (opacity_image.cpu().numpy() * 255).astype(np.uint8)
            opacity_image = np.hstack(opacity_image)

            image = np.vstack((rgb_image, opacity_image))
            image = Image.fromarray(image)
            # image.show()

            video.add(rgb_image)
            for _ in range(4):
                s, _ = model(s)
            camera.rotateY(4.0)
