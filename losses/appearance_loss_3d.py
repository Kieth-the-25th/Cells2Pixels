import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from PIL import Image


class AppearanceLoss3D(torch.nn.Module):
    """3D appearance (style) loss for volumetric data.

    Operates on tensors of shape ``(B, C, D, H, W)`` using a pre-trained
    R3D-18 backbone to extract volumetric features, then computes an
    optimal-transport + auto-correlation loss on those features.

    Designed as a drop-in 3D analogue of ``AppearanceLoss`` (the 2D version).
    """

    def __init__(self,
                 target_volume_size=(64, 64, 64),
                 patch_size=(32, 32, 32),
                 num_scales=1,
                 target_volume_path=None,
                 total_channels=1,
                 output_channels=None,
                 r3d_layers=(1, 2, 3, 4),
                 include_volume_as_feature=False,
                 ot_loss_weight=1.0,
                 ac_loss_weight=0.0,
                 ac_loss_kwargs=None,
                 device='cuda:0'):
        """
        :param target_volume_size: (D, H, W) for loading / resizing target.
        :param patch_size: (d, h, w) crop size for feature extraction.
        :param num_scales: multi-scale pyramid levels.
        :param target_volume_path: path to a numpy ``.npy`` volume, or a dict
                                   mapping keys to paths.
        :param total_channels: total channels expected in the rendered volume.
        :param output_channels: dict mapping target key -> list of channel indices.
        :param r3d_layers: which R3D stages to extract features from
                           (1,2,3,4 correspond to layer1 … layer4).
        :param include_volume_as_feature: prepend the normalised volume to
                                          the feature list.
        :param ot_loss_weight: weight for the optimal-transport loss term.
        :param ac_loss_weight: weight for the auto-correlation loss term.
        :param ac_loss_kwargs: kwargs forwarded to ``AutoCorrelationLoss3D``.
        :param device: torch device.
        """
        super().__init__()

        self.target_volume_size = target_volume_size
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.total_channels = total_channels
        self.r3d_layers = tuple(r3d_layers)
        self.include_volume_as_feature = include_volume_as_feature
        self.ot_loss_weight = ot_loss_weight
        self.ac_loss_weight = ac_loss_weight
        self.device = device

        if output_channels is None:
            output_channels = {"default": list(range(total_channels))}
        self.output_channels = output_channels

        # --- feature backbone: R3D-18 pre-trained on Kinetics ---
        r3d = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        )
        # Extract the per-stage modules we need
        self.stem = r3d.stem      # Conv3d + BN + ReLU + MaxPool
        self.layer1 = r3d.layer1  # 2 BasicBlocks
        self.layer2 = r3d.layer2
        self.layer3 = r3d.layer3
        self.layer4 = r3d.layer4
        self._stage_modules = [self.layer1, self.layer2, self.layer3, self.layer4]

        self._normalise_mean = torch.tensor(
            [0.43216, 0.394666, 0.37645], device=device
        ).view(1, 3, 1, 1, 1)
        self._normalise_std = torch.tensor(
            [0.22803, 0.22145, 0.216989], device=device
        ).view(1, 3, 1, 1, 1)

        self._load_target(target_volume_path)

        self.ot_loss_fn = OptimalTransportLoss3D(n_samples=1024, device=device)
        self.ac_loss_fn = AutoCorrelationLoss3D(**(ac_loss_kwargs or {}))

    # ---- feature extraction ------------------------------------------------

    def normalise(self, x):
        return (x - self._normalise_mean) / self._normalise_std

    def get_r3d_features(self, x, flatten=False):
        """
        :param x: (B, 3, D, H, W)  normalised input volume.
        :param flatten: if True, each returned tensor is (B, C, N) where N=D*H*W.
        :return: list of feature tensors from selected R3D stages.
        """
        B, C, D, H, W = x.shape
        features = []

        if self.include_volume_as_feature:
            features.append(x.reshape(B, C, D * H * W) if flatten else x)

        x = self.stem(x)   # -> (B, 64, D//2, H//2, W//2)  (approx)
        for idx, module in enumerate(self._stage_modules):
            x = module(x)
            stage = idx + 1
            if stage in self.r3d_layers:
                feat = x
                if flatten:
                    b, c, d, h, w = feat.shape
                    feat = feat.reshape(b, c, d * h * w)
                features.append(feat)
        return features

    def extract_features(self, target_volume):
        """Multi-scale feature pyramid for a target volume.

        :param target_volume: (n_targets, 3, D, H, W)
        :return: list of feature lists, one per scale.
        """
        pyramid = []
        d, h, w = self.patch_size
        for i in range(self.num_scales):
            scale = 2 ** i
            size = (d * scale, h * scale, w * scale)
            x = F.interpolate(target_volume, size=size, mode='trilinear',
                              align_corners=False)
            pyramid.append(self.get_r3d_features(x))
        return pyramid

    # ---- target loading ---------------------------------------------------

    @torch.no_grad()
    def _load_target(self, path):
        if path is None:
            # Dummy target — user is expected to set it externally or
            # this is just a feature extractor.
            self.target_volume = None
            self.target_features = None
            return

        if isinstance(path, dict):
            vols = []
            for key in sorted(path):
                vol = self._load_single(path[key])
                vols.append(vol)
            self.target_volume = torch.cat(vols, dim=0)
        else:
            self.target_volume = self._load_single(path)

        self.target_volume = self.target_volume.to(self.device)
        self.target_features = self.extract_features(self.target_volume)

    def _load_single(self, path):
        vol = np.load(path).astype(np.float32)  # (D, H, W) or (D, H, W, C)
        if vol.ndim == 3:
            vol = vol[None]                     # -> (1, D, H, W)
        else:
            vol = vol.transpose(3, 0, 1, 2)     # -> (C, D, H, W)
        vol = torch.from_numpy(vol)
        # Resize to target size
        vol = vol.unsqueeze(0)  # (1, C, D, H, W)
        vol = F.interpolate(vol, size=self.target_volume_size,
                            mode='trilinear', align_corners=False)
        # Repeat to 3 channels if needed
        if vol.shape[1] == 1:
            vol = vol.repeat(1, 3, 1, 1, 1)
        elif vol.shape[1] > 3:
            vol = vol[:, :3]
        return vol

    # ---- forward -----------------------------------------------------------

    def forward(self, input_dict, return_summary=True):
        """
        :param input_dict: dict with key ``'rendered_volume'``
                           ``(B, C, D, H, W)`` in [0, 1].
        :param return_summary: if True, return a PIL summary image.
        :return: ``(loss, loss_log, summary)``
        """
        rendered = input_dict['rendered_volume']  # (B, C, D, H, W)
        B, C, D, H, W = rendered.shape
        assert C == self.total_channels, \
            f"Expected {self.total_channels} channels, got {C}"

        loss = 0.0
        ot_loss_val = 0.0
        ac_loss_val = 0.0
        summary = None

        for scale in range(self.num_scales):
            x = rendered
            if scale != 0:
                # Multi-scale: random crops
                ds, hs, ws = D // (2 ** scale), H // (2 ** scale), W // (2 ** scale)
                n_crops = max(1, int(2.0 * scale))
                crops = []
                for _ in range(n_crops):
                    d_off = torch.randint(0, D - ds + 1, (1,)).item()
                    h_off = torch.randint(0, H - hs + 1, (1,)).item()
                    w_off = torch.randint(0, W - ws + 1, (1,)).item()
                    crops.append(x[:, :, d_off:d_off + ds, h_off:h_off + hs, w_off:w_off + ws])
                x = torch.cat(crops, dim=0)

            # Split channels per output key
            xs = []
            for key in sorted(self.output_channels):
                oc = self.output_channels[key]
                ch = x[:, oc]
                if len(oc) == 1:
                    ch = ch.repeat(1, 3, 1, 1, 1)
                xs.append(ch)

            n_targets = len(xs)
            generated = torch.stack(xs, dim=1)  # (B, n_targets, 3, D, H, W)

            if return_summary and scale == 0:
                with torch.no_grad():
                    images = self._make_summary(generated, self.target_volume)
                    summary = {"images": images}

            generated = generated.view(-1, 3, D, H, W)          # (B*n_targets, 3, D, H, W)
            # Resize to patch size
            generated = F.interpolate(
                generated, size=self.patch_size,
                mode='trilinear', align_corners=False
            )

            gen_features = self.get_r3d_features(generated)
            tgt_features = self.target_features[scale]

            ot_loss_val += self.ot_loss_fn(tgt_features, gen_features)
            if scale == 0 and self.ac_loss_weight > 0.0:
                ac_loss_val += self.ac_loss_fn(tgt_features, gen_features)

        ot_loss_val = ot_loss_val / self.num_scales
        loss = self.ot_loss_weight * ot_loss_val + self.ac_loss_weight * ac_loss_val

        loss_log = {
            k: ot_loss_val[i]
            for i, k in enumerate(sorted(self.output_channels))
        }
        if self.ac_loss_weight > 0.0:
            for i, k in enumerate(sorted(self.output_channels)):
                loss_log[f"AC-{k}"] = ac_loss_val[i]

        return loss, loss_log, summary

    # ---- summary helpers ---------------------------------------------------

    @torch.no_grad()
    def _make_summary(self, generated, target):
        """Create a 2D tiled summary from middle slices."""
        B, nT, _, D, H, W = generated.shape
        # Gather first 4 batch items and all targets
        nB = min(B, 4)
        nT = min(nT, 6)
        mid_d = D // 2
        mid_h = H // 2
        mid_w = W // 2

        rows = []
        for b in range(nB):
            row = []
            for t in range(nT):
                vol = generated[b, t]
                # Concatenate three orthogonal slices
                slice_d = vol[:, mid_d, :, :]       # (3, H, W)
                slice_h = vol[:, :, mid_h, :]       # (3, D, W)
                slice_w = vol[:, :, :, mid_w]       # (3, D, H)
                slices = []
                for s in (slice_d, slice_h, slice_w):
                    s = s.clamp(0, 1).cpu().numpy()
                    s = (s.transpose(1, 2, 0) * 255).astype(np.uint8)
                    slices.append(Image.fromarray(s))
                # Stack slices vertically
                col = np.vstack(slices)
                row.append(col)
            if target is not None:
                t = target[0].clamp(0, 1).cpu().numpy()
                t_d = (t[:, mid_d, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
                t_h = (t[:, :, mid_h, :].transpose(1, 2, 0) * 255).astype(np.uint8)
                t_w = (t[:, :, :, mid_w].transpose(1, 2, 0) * 255).astype(np.uint8)
                tgt_col = np.vstack([Image.fromarray(x) for x in (t_d, t_h, t_w)])
                row = [tgt_col] + row
            rows.append(np.hstack(row))
        return Image.fromarray(np.vstack(rows))


class OptimalTransportLoss3D(torch.nn.Module):
    """Optimal-transport style loss on volumetric feature sets.

    Dimension-agnostic — operates on flattened feature sets ``(B, N, C)``.
    Adapted from ``OptimalTransportLoss`` in the 2D appearance loss.
    """

    def __init__(self, n_samples=1024, device='cuda:0'):
        super().__init__()
        self.n_samples = n_samples
        self.device = device
        self.color_transform = torch.tensor(
            [[0.577350, 0.577350, 0.577350],
             [-0.577350, 0.788675, -0.211325],
             [-0.577350, -0.211325, 0.788675]],
            device=device, requires_grad=False
        )

    def rgb_to_yuv(self, rgb):
        return torch.einsum('bnc,ck->bnk', rgb, self.color_transform)

    def color_matching_loss(self, x, y):
        x_yuv = self.rgb_to_yuv(x)
        y_yuv = self.rgb_to_yuv(y)
        dist = (self.pairwise_distances_l2(x_yuv, y_yuv) +
                self.pairwise_distances_cos(x_yuv, y_yuv))
        return torch.max(dist.min(1)[0].mean(dim=1),
                         dist.min(2)[0].mean(dim=1))

    @staticmethod
    def pairwise_distances_l2(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True) ** 2
        y_t = y.transpose(1, 2)
        y_norm = torch.norm(y_t, dim=1, keepdim=True) ** 2
        cross = torch.matmul(x, y_t)
        dist = x_norm + y_norm - 2.0 * cross
        return torch.clamp(dist, 1e-5, 1e5) / x.size(2)

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)
        y_t = y.transpose(1, 2)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)
        return 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)

    @staticmethod
    def style_loss(x, y, metric="cos"):
        if metric == "cos":
            dist = OptimalTransportLoss3D.pairwise_distances_cos(x, y)
        else:
            dist = OptimalTransportLoss3D.pairwise_distances_l2(x, y)
        return torch.max(dist.min(1)[0].mean(dim=1),
                         dist.min(2)[0].mean(dim=1))

    @staticmethod
    def moment_loss(x, y):
        mu_x = torch.mean(x, 1, keepdim=True)
        mu_y = torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))
        x_c = x - mu_x
        y_c = y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)
        return mu_diff + torch.abs(x_cov - y_cov).mean(dim=(1, 2))

    def forward(self, target_features, generated_features):
        """
        :param target_features:    list of tensors per R3D stage,
                                   each ``(n_targets, C, D*H*W)``
        :param generated_features: list of tensors per R3D stage,
                                   each ``(B * n_targets, C, D*H*W)``
        :return: per-target loss ``(n_targets,)``
        """
        loss = 0.0
        for i, (y, x) in enumerate(zip(target_features, generated_features)):
            n_targets, c_y, n_y = y.shape
            B, c_x, n_x = x.shape
            batch_size = B // n_targets
            assert batch_size * n_targets == B

            y = y.repeat(batch_size, 1, 1)
            n_samples = min(n_x, n_y, self.n_samples)

            idx_x = torch.argsort(torch.rand(B, 1, n_x, device=x.device),
                                  dim=-1)[..., :n_samples]
            x_samp = x.gather(-1, idx_x.expand(B, c_x, n_samples))

            idx_y = torch.argsort(torch.rand(B, 1, n_y, device=y.device),
                                  dim=-1)[..., :n_samples]
            y_samp = y.gather(-1, idx_y.expand(B, c_y, n_samples))

            x_samp = x_samp.transpose(1, 2)  # (B, n_samples, C)
            y_samp = y_samp.transpose(1, 2)

            if i == 0 and c_x == c_y == 3:
                layer_loss = self.color_matching_loss(x_samp, y_samp)
            else:
                layer_loss = (self.style_loss(x_samp, y_samp) +
                              self.moment_loss(x_samp, y_samp))
            loss += layer_loss

        loss = loss.view(batch_size, n_targets).mean(dim=0)
        return loss


class AutoCorrelationLoss3D(torch.nn.Module):
    """3D auto-correlation loss on volumetric feature maps.

    Uses 3D FFT to compute spatial auto-correlation maps per feature channel,
    then compares target and generated maps with an L1 / MSE loss.

    Operates on feature tensors of shape ``(B, C, D, H, W)``.
    """

    def __init__(self,
                 layers=(0, 1),
                 reduction="l1",
                 normalize=True,
                 exclude_zero_lag=True,
                 center_crop=None,
                 layer_weights=None,
                 eps=1e-6,
                 fft_norm="backward"):
        super().__init__()
        assert reduction in ("l1", "mse")
        self.layers = tuple(layers)
        self.reduction = reduction
        self.normalize = normalize
        self.exclude_zero_lag = exclude_zero_lag
        self.center_crop = center_crop
        self.layer_weights = layer_weights
        self.eps = eps
        self.fft_norm = fft_norm

    @staticmethod
    def _roll_center(x):
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        return torch.roll(x, shifts=(D // 2, H // 2, W // 2), dims=(-3, -2, -1))

    @staticmethod
    def _center_crop(x, half):
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        if isinstance(half, int):
            hd = hh = hw = half
        else:
            hd, hh, hw = half
        top_d = max(D // 2 - hd, 0)
        top_h = max(H // 2 - hh, 0)
        top_w = max(W // 2 - hw, 0)
        dd = min(2 * hd + 1, D)
        dh = min(2 * hh + 1, H)
        dw = min(2 * hw + 1, W)
        return x[..., top_d:top_d + dd, top_h:top_h + dh, top_w:top_w + dw]

    def _autocorr_map(self, feat):
        """Compute channel-aggregated 3D auto-correlation map.

        :param feat: (B, C, D, H, W)
        :return: (B, D, H, W)
        """
        B, C, D, H, W = feat.shape
        f = feat - feat.mean(dim=(-3, -2, -1), keepdim=True)

        # 3D real FFT
        Fz = torch.fft.rfftn(f, dim=(-3, -2, -1), norm=self.fft_norm)
        power = (Fz.real ** 2 + Fz.imag ** 2).sum(dim=1)  # (B, D, H, W//2+1)

        ac = torch.fft.irfftn(power, s=(D, H, W), dim=(-3, -2, -1),
                              norm=self.fft_norm)  # (B, D, H, W)

        if self.normalize:
            ac = ac / ac[..., 0:1, 0:1, 0:1].clamp_min(self.eps)

        if self.exclude_zero_lag:
            ac[..., 0, 0, 0] = 0

        if self.center_crop is not None:
            rolled = self._roll_center(ac)
            cropped = self._center_crop(rolled, self.center_crop)
            ac = self._roll_center(cropped)

        return ac

    def _layer_loss(self, gen_ac, tgt_ac):
        n_targets = tgt_ac.shape[0]
        B = gen_ac.shape[0]
        batch_size = B // n_targets
        tgt_rep = tgt_ac.repeat(batch_size, 1, 1, 1)

        if self.reduction == "l1":
            d = torch.abs(gen_ac - tgt_rep).mean(dim=(-3, -2, -1))
        else:
            d = ((gen_ac - tgt_rep) ** 2).mean(dim=(-3, -2, -1))

        return d.view(batch_size, n_targets).mean(dim=0)

    def forward(self, target_features, generated_features):
        """
        :param target_features:    list of tensors per R3D stage,
                                   each ``(n_targets, C, D, H, W)``
        :param generated_features: list of tensors per R3D stage,
                                   each ``(B, C, D, H, W)``
        :return: ``(n_targets,)``
        """
        loss_vec = None
        for i, (t, g) in enumerate(zip(target_features, generated_features)):
            if self.layers is not None and i not in self.layers:
                continue

            with torch.no_grad():
                t_ac = self._autocorr_map(t)
            g_ac = self._autocorr_map(g)

            li = self._layer_loss(g_ac, t_ac)

            w = 1.0
            if self.layer_weights is not None:
                try:
                    w = float(self.layer_weights[self.layers.index(i)])
                except Exception:
                    pass
            loss_vec = li * w if loss_vec is None else loss_vec + li * w

        if loss_vec is None:
            raise ValueError("No layers matched in AutoCorrelationLoss3D")
        return loss_vec


if __name__ == '__main__':
    # Quick smoke test with random data
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running smoke test on {device}")

    loss_fn = AppearanceLoss3D(
        target_volume_size=(32, 32, 32),
        patch_size=(16, 16, 16),
        total_channels=3,
        output_channels={"RGB": [0, 1, 2]},
        device=device,
    )

    dummy_vol = torch.rand(2, 3, 32, 32, 32, device=device)
    dummy_input = {"rendered_volume": dummy_vol}

    loss, loss_log, summary = loss_fn(dummy_input, return_summary=True)
    print("Loss:", loss.item())
    print("Loss log:", loss_log)
    if summary:
        print("Summary image:", summary["images"].size)
        summary["images"].show()
