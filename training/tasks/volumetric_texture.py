import copy

import numpy as np
import torch
from tqdm import tqdm

from losses.loss import Loss
from losses.appearance_loss_3d import AppearanceLoss3D
from models.nca3d import VNCA
from models.siren import Siren
from training.common import (
    TestOptions,
    device_config,
    load_checkpoint_pair,
    load_graft_if_configured,
    make_grad_scaler,
    normalize_model_grads,
    optimizer_scheduler_step,
    precision_from_config,
    save_checkpoint,
    set_seed,
)
from training.tasks.base import BaseTask
from utils.misc import autocast_context, process_output_channels


class VolumetricTextureTask(BaseTask):
    """
    texture_2d but 3 dimensions
    """

    def _build(self, load: bool = False):
        precision = precision_from_config(self.config)
        nca_kwargs = device_config(self.config["nca"]["nca_kwargs"], self.device)
        nca_kwargs["precision"] = precision
        model = VNCA(**nca_kwargs).to(self.device)
        total_channels, output_channels = process_output_channels(
            self.config["num_channels"]
        )
        nca_output_dim = model.channels
        if self.config["nca"]["output_type"] == "z":
            nca_output_dim *= model.perception_kernels
        # For volumetric texture we output density + color channels, same as
        # the volumetric renderer path, so sh_degree=0 is used.
        sh_degree = self.config.get("sh_degree", 0)
        output_dim = 1 + total_channels * ((sh_degree + 1) ** 2)
        siren = Siren(
            in_features=nca_output_dim,
            coord_dim=3,
            out_features=output_dim,
            **self.config["siren"],
        ).to(self.device)
        if load:
            load_checkpoint_pair(self.config, model, siren, device=self.device)
        return model, siren, precision, output_channels, total_channels

    def _decode_grid(
        self, nca_state: torch.Tensor, siren: Siren, grid_size: int
    ) -> torch.Tensor:
        """Decode the full 3D NCA grid through the Siren into a volume.

        Args:
            nca_state: (B, C, D, H, W)  cell states.
            siren: The Siren decoder.
            grid_size: spatial resolution (same for D/H/W).

        Returns:
            Volume tensor (B, C_rgb, D, H, W) where C_rgb excludes density.
        """
        B, C, D, H, W = nca_state.shape
        N = D * H * W
        states_flat = nca_state.reshape(B, N, C)

        lin = torch.linspace(-1.0, 1.0, grid_size, device=self.device)
        z_grid, y_grid, x_grid = torch.meshgrid(lin, lin, lin, indexing="ij")
        coords = torch.stack([x_grid, y_grid, z_grid], dim=-1).reshape(1, N, 3)
        coords = coords.expand(B, -1, -1)

        output = siren(states_flat, coords)  # (B, N, 1 + C_rgb)
        # First channel = density (sigma), remaining = colors
        rgb = output[:, :, 1:]  # (B, N, C_rgb)
        C_rgb = rgb.shape[-1]
        return rgb.reshape(B, C_rgb, D, H, W)

    def train(self) -> None:
        set_seed(self.config.get("seed", 42))
        model, siren, precision, output_channels, total_channels = self._build()
        self._log_counts(model, siren, "VNCA")
        load_graft_if_configured(self.config, "nca", model, siren, self.device)

        # Wire the 3D appearance loss into the composite Loss
        self.config["loss"]["appearance_loss_3d_kwargs"] = {
            "target_volume_path": self.config["loss"].get("target_volume_path"),
            "target_volume_size": self.config["loss"].get(
                "target_volume_size", [48, 48, 48]
            ),
            "patch_size": self.config["loss"].get("patch_size", [32, 32, 32]),
            "num_scales": self.config["loss"].get("num_scales", 1),
            "total_channels": total_channels,
            "output_channels": output_channels,
        }
        with torch.no_grad():
            loss_fn = Loss(**device_config(self.config["loss"], self.device))
            grid_size = self.config["nca"]["grid_size"]
            D, H, W = grid_size

        train_cfg = self.config["train"]
        batch_size = train_cfg["batch_size"]
        accumulation_steps = (
            train_cfg["virtual_batch_size"] + batch_size - 1
        ) // batch_size

        for repetition in range(train_cfg["num_repetitions"]):
            with torch.no_grad():
                pool = model.seed(train_cfg["pool_size"], *grid_size)

            parameters = list(model.parameters()) + list(siren.parameters())
            optimizer, scheduler = self._optimizer(parameters)
            scaler = make_grad_scaler(self.device, precision)
            accumulation_counter = 0

            total_steps = train_cfg["epochs"] * accumulation_steps
            for epoch in tqdm(
                range(total_steps),
                desc=f"Repetition {repetition + 1}/{train_cfg['num_repetitions']}",
            ):
                log_step = (
                    epoch + repetition * train_cfg["epochs"] * accumulation_steps
                ) // accumulation_steps
                virtual_epoch = epoch // accumulation_steps

                with torch.no_grad():
                    batch_idx = np.random.choice(
                        len(pool), batch_size, replace=False
                    )
                    x = pool[batch_idx]
                    if (
                        virtual_epoch % train_cfg["inject_seed_interval"] == 0
                        and accumulation_counter == 0
                    ):
                        x[:1] = model.seed(1, *grid_size)

                step_n = np.random.randint(*train_cfg["step_range"])
                z = None
                with autocast_context(self.device, precision):
                    for st in range(step_n):
                        x, z = model(x)
                pool[batch_idx] = x.detach()

                x_render = (
                    x if self.config["nca"]["output_type"] == "s" else z
                ).to(torch.float32)

                volume = self._decode_grid(x_render, siren, D)  # (B, C, D, H, W)

                input_dict = {
                    "rendered_volume": volume,
                    "nca_state": x,
                }
                return_summary = (
                    log_step % train_cfg["summary_interval"] == 0
                    and accumulation_counter == 0
                )
                loss, loss_log, summary = loss_fn(
                    input_dict, return_summary=return_summary
                )
                accumulation_counter += 1

                if return_summary and summary:
                    for key, img in summary.items():
                        self._save_logged_image(key, img, log_step)

                if precision == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if accumulation_counter == accumulation_steps:
                    with torch.no_grad():
                        normalize_model_grads(model)
                        optimizer_scheduler_step(
                            optimizer, scheduler, scaler, precision
                        )
                        optimizer.zero_grad()
                        loss_fn.update_loss_weights(loss_log, log_step)
                    accumulation_counter = 0
                    self.logger.log_metrics(loss_log, step=log_step)

            save_checkpoint(
                self.config, model, siren, suffix=f"_{repetition + 1}"
            )
        save_checkpoint(self.config, model, siren)

    @torch.no_grad()
    def test(self, options: TestOptions) -> None:
        test_config = copy.deepcopy(self.config)
        test_config["precision"] = "float32"
        self.config = test_config
        model, siren, _, _, _ = self._build(load=True)
        output_dir = self._output_dir(options)
        grid_size = self.config["nca"]["grid_size"]
        D, H, W = grid_size

        x = model.seed(1, *grid_size)

        z = None
        for _ in tqdm(range(options.steps), desc="Test rollout"):
            x, z = model(x)

        x_render = (
            x if self.config["nca"]["output_type"] == "s" else z
        ).to(torch.float32)

        volume = self._decode_grid(x_render, siren, D)  # (1, C, D, H, W)
        vol_np = volume[0].cpu().numpy()  # (C, D, H, W)

        # Save as (D, H, W, C) for intuitive numpy loading
        vol_np = vol_np.transpose(1, 2, 3, 0)  # (D, H, W, C)

        path = output_dir / f"{self.config['experiment_name']}.npy"
        np.save(str(path), vol_np)
        print(f"Saved 3D texture → {path}")

        # Also save orthogonal slice images for quick preview
        from PIL import Image

        mid = [D // 2, H // 2, W // 2]
        slices = []
        for ax, idx in enumerate(mid):
            if ax == 0:
                s = vol_np[idx, :, :, :3]
            elif ax == 1:
                s = vol_np[:, idx, :, :3]
            else:
                s = vol_np[:, :, idx, :3]
            s = np.clip(s, 0.0, 1.0)
            slices.append(Image.fromarray((s * 255).astype(np.uint8)))

        # Stack slices vertically
        stacked = np.vstack(slices)
        Image.fromarray(stacked).save(
            output_dir / f"{self.config['experiment_name']}_slices.png"
        )
