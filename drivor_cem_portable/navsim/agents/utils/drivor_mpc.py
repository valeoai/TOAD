"""Shared helpers for using the DrivoR scorer as a CEM/MPC objective inside
other navsim agents (gtrs_dense, gtrs_aug, ztrs, pad, rap_dino, sparsedrive).

The pattern in each host model:

    from navsim.agents.utils import drivor_mpc

    # in __init__ (after building the host model):
    self._drivor_model = None
    if config.use_drivor_score:
        self._drivor_model = drivor_mpc.load_drivor_model(
            config.drivor_config_path,
            config.drivor_ckpt_path,
            overrides=config.drivor_overrides,
        )

    # in cem_refine (or wherever the host model picks its final traj):
    if self._drivor_model is not None:
        score_fn = drivor_mpc.make_drivor_score_fn(
            self._drivor_model,
            features,
            subsample_indices=config.drivor_subsample_indices,
        )

In each feature builder, build the DrivoR-compatible camera + ego tensors:

    if config.use_drivor_score:
        features["drivor_image"]      = drivor_mpc.build_drivor_image(agent_input, drivor_cfg)
        features["drivor_ego_status"] = drivor_mpc.build_drivor_ego_status(agent_input)
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# Cameras DrivoR's feature builder considers, in the same order it stacks them.
_DRIVOR_CAM_KEYS = (
    "cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2",
)


def load_drivor_model(
    config_path: str,
    ckpt_path: str = "",
    overrides: Optional[Any] = None,
) -> nn.Module:
    """Build a DrivoRModel from a yaml + optional overrides, then load weights.

    Returned model is in eval() mode with all parameters frozen — it's used
    purely as a scorer.
    """
    from omegaconf import OmegaConf
    from navsim.agents.drivoR.drivor_model import DrivoRModel

    drivor_cfg = OmegaConf.load(config_path).config
    if overrides:
        drivor_cfg = OmegaConf.merge(drivor_cfg, OmegaConf.create(dict(overrides)))
    drivor_model = DrivoRModel(drivor_cfg)

    if ckpt_path:
        map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(ckpt_path, map_location=map_loc)["state_dict"]
        clean = {}
        for k, v in state_dict.items():
            for prefix in ("agent._drivor_model.", "_drivor_model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            clean[k] = v
        msg = drivor_model.load_state_dict(clean, strict=False)
        print("[drivor_mpc] loaded DrivoR scorer checkpoint:", msg)

    drivor_model.eval()
    for p in drivor_model.parameters():
        p.requires_grad_(False)
    return drivor_model


def make_drivor_score_fn(
    drivor_model: nn.Module,
    features: dict,
    subsample_indices: Optional[Iterable[int]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a (B, N, P, 3) -> (B, N) score function backed by DrivoR.

    The DrivoR scorer was trained on 8-pose, 0.5s-interval trajectories. If
    the host model produces 40-pose trajectories (e.g. gtrs_*), pass
    ``subsample_indices=(4, 9, ..., 39)`` to pick the 8 timestamps DrivoR
    expects. Pass ``None`` (or an empty iterable) when the host model
    already produces 8 poses (pad / rap / sparsedrive).

    scene_features and ego_token are computed ONCE per call (not per
    score_fn invocation) — the inner closure just reuses them.
    """
    drivor = drivor_model
    drivor.eval()  # keep dinov2 grid_mask / dropout disabled
    device = next(drivor.parameters()).device
    dtype = next(drivor.parameters()).dtype

    ego_status_raw = features["drivor_ego_status"].to(device=device, dtype=dtype)
    if drivor._config.full_history_status:
        ego_in = ego_status_raw.flatten(-2)
    else:
        ego_in = ego_status_raw[:, -1]
    ego_token = drivor.hist_encoding(ego_in)[:, None]
    B = ego_in.shape[0]

    scene_chunks = []
    if drivor.num_cams > 0:
        img = features["drivor_image"].to(device=device, dtype=dtype)
        scene_tokens = drivor.scene_embeds.repeat(B, 1, 1, 1)
        scene_chunks.append(drivor.image_backbone(img, scene_tokens))
    if drivor.num_lidar > 0:
        lidar = features["drivor_lidar"].to(device=device, dtype=dtype)
        scene_tokens = drivor.lidar_scene_embeds.repeat(B, 1, 1, 1)
        scene_chunks.append(drivor.lidar_backbone(lidar, scene_tokens))
    scene_features = torch.cat(scene_chunks, dim=1)

    sub_idx = None
    if subsample_indices is not None and len(tuple(subsample_indices)) > 0:
        sub_idx = torch.tensor(
            tuple(subsample_indices), device=device, dtype=torch.long
        )

    # DrivoR's pos_embed is Linear(poses_num * 3, d_ffn). If the host model
    # produces a different horizon and no explicit subsample is given, fall
    # back to the first `drivor_poses` poses (matches DrivoR's training
    # window: t=0.5s..4s on a 0.5s grid).
    drivor_poses = drivor.poses_num

    def score_fn(traj: torch.Tensor) -> torch.Tensor:
        traj_d = traj.to(device=device, dtype=dtype)
        if sub_idx is not None:
            traj_d = traj_d.index_select(-2, sub_idx)
        elif traj_d.shape[-2] != drivor_poses:
            traj_d = traj_d[..., :drivor_poses, :]
        return drivor._score_trajectories(traj_d, scene_features, ego_token)

    return score_fn


def build_drivor_image(agent_input, drivor_cfg) -> torch.Tensor:
    """Build the per-camera tensor DrivoR's ImgEncoder expects.

    Mirrors drivor_features.py: pick cameras whose config list is non-empty,
    resize each to ``drivor_cfg["image_size"] = (W, H)``, then ImageNet
    normalize. Returns shape (N_cams, 3, H, W).
    """
    image_size = tuple(drivor_cfg["image_size"])
    cameras = agent_input.cameras[-1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    images = []
    for cam_key in _DRIVOR_CAM_KEYS:
        if cam_key not in drivor_cfg or len(drivor_cfg[cam_key]) == 0:
            continue
        cam = getattr(cameras, cam_key)
        if cam.image is None:
            continue
        im = Image.fromarray(cam.image).resize(image_size)
        im = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
        images.append(torch.from_numpy(im).permute(2, 0, 1))
    return torch.stack(images)


def build_drivor_ego_status(agent_input) -> torch.Tensor:
    """Full-history ego status in DrivoR layout: (T_hist, 11) =
    [pose(3), vel(2), accel(2), driving_command(4)] per timestep."""
    ego_features = []
    for ego_status in agent_input.ego_statuses:
        if ego_status is None:
            continue
        pose = torch.tensor(ego_status.ego_pose, dtype=torch.float32)
        vel = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
        acc = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
        cmd = torch.tensor(ego_status.driving_command, dtype=torch.float32)
        ego_features.append(torch.cat([pose, vel, acc, cmd], dim=-1))
    return torch.stack(ego_features)


def load_drivor_cfg(config_path: str, overrides: Optional[Any] = None):
    """Load + merge a DrivoR yaml config — used by feature builders that
    need to know image_size / camera list before the model is constructed."""
    from omegaconf import OmegaConf
    drivor_cfg = OmegaConf.load(config_path).config
    if overrides:
        drivor_cfg = OmegaConf.merge(
            drivor_cfg, OmegaConf.create(dict(overrides))
        )
    return drivor_cfg
