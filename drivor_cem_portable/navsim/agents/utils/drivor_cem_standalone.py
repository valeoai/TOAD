"""Standalone DrivoR + CEM(MPC) refinement.

This module wires the DrivoR scorer (`navsim/agents/utils/drivor_mpc.py`) into
the model-agnostic Cross-Entropy-Method optimizer
(`navsim/agents/utils/cem_optimize` in `cem_utils.py`). It is self-contained:
it loads DrivoR from the default config/checkpoint below, wraps the DrivoR
model as a CEM `score_fn`, and runs CEM over a set of seed trajectories with the
DrivoR scorer plus the closed-form comfort cost and the anchor regularizer.

The ablation knobs in `cem_utils.cem_optimize` are intentionally left at their
baseline values — we only use scorer + comfort + anchor costs.

Typical use:

    from navsim.agents.utils import drivor_cem_standalone as dcs

    model = dcs.load_drivor()                      # default cfg/ckpt + seed=2
    features = {
        "drivor_image":      <(B, N_cams, 3, H, W)>,   # build_drivor_image
        "drivor_ego_status": <(B, T_hist, 11)>,        # build_drivor_ego_status
    }
    out = dcs.cem_refine_with_drivor(
        model, features,
        proposals=<(B, N_p, P, 3)>,   # seed trajectories
        best_traj=<(B, P, 3)>,        # argmax / anchor trajectory
    )
    refined = out["trajectory"]       # (B, P, 3)
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional

import torch

from navsim.agents.utils import cem_utils
from navsim.agents.utils import drivor_mpc


# --------------------------------------------------------------------------- #
# Default DrivoR config / checkpoint / overrides (from the requested CLI).
# --------------------------------------------------------------------------- #

def _root() -> str:
    """Portable-package root.

    Resolved from this file's location
    (``<root>/navsim/agents/utils/drivor_cem_standalone.py``), so the whole
    `drivor_cem_portable/` folder runs from anywhere without env setup, and is
    unaffected by the $NAVSIM_DEVKIT_ROOT that the main NAVSIM workflow exports.

    Set $DRIVOR_CEM_ROOT to point the bundle at a different checkout instead.
    """
    root = os.environ.get("DRIVOR_CEM_ROOT", "")
    if not root:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
    return root


def _default_config_path() -> str:
    return os.path.join(
        _root(), "navsim", "planning", "script", "config", "common", "agent",
        "drivoR.yaml",
    )


def _default_ckpt_path() -> str:
    """Path to the bundled DrivoR checkpoint (existence is checked at load time)."""
    return os.path.join(_root(), "checkpoints", "epoch_29_54.6.ckpt")


DEFAULT_DRIVOR_CONFIG_PATH: str = _default_config_path()

DEFAULT_DRIVOR_CKPT_PATH: str = _default_ckpt_path()

# Mirrors `agent.config.drivor_overrides.*` from the requested launch command.
DEFAULT_DRIVOR_OVERRIDES: dict = {
    "proposal_num": 64,
    "refiner_ls_values": 0.0,
    "image_backbone": {"focus_front_cam": False},
    "one_token_per_traj": True,
    "refiner_num_heads": 1,
    "tf_d_model": 256,
    "tf_d_ffn": 1024,
    "area_pred": False,
    "agent_pred": False,
    "ref_num": 4,
    "noc": 10,
    "dac": 13,
    "ddc": 6,
    "ttc": 14,
    "ep": 15,
    "comfort": 2,
}

DEFAULT_SEED: int = 2

# CEM defaults requested for this standalone runner.
DEFAULT_CEM_ITERATIONS: int = 5
DEFAULT_CEM_SAMPLES: int = 64
DEFAULT_CEM_DT: float = 0.5


# --------------------------------------------------------------------------- #
# 1) Load DrivoR.
# --------------------------------------------------------------------------- #

def _absolutize_backbone_weights(overrides: dict) -> dict:
    """Point the DINOv2 backbone at the bundled weights with an absolute path.

    The yaml's ``image_backbone.model_weights`` is a cwd-relative path that
    ``timm.create_model`` reads directly, so it only resolves when run from the
    folder root. If the bundled ``weights/`` directory exists we rewrite it to
    an absolute path so the model loads regardless of the working directory.
    """
    bundled = os.path.join(
        _root(), "weights", "vit_small_patch14_reg4_dinov2.lvd142m",
        "model.safetensors",
    )
    if not os.path.exists(bundled):
        return overrides
    ib = dict(overrides.get("image_backbone", {}) or {})
    ib["model_weights"] = bundled
    overrides["image_backbone"] = ib
    return overrides


def load_drivor(
    config_path: str = DEFAULT_DRIVOR_CONFIG_PATH,
    ckpt_path: str = DEFAULT_DRIVOR_CKPT_PATH,
    overrides: Optional[dict] = None,
    seed: Optional[int] = DEFAULT_SEED,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Build + load a frozen, eval-mode DrivoR model used purely as a scorer.

    Args:
        config_path: path to drivoR.yaml (with a top-level `config:` block).
        ckpt_path:   DrivoR lightning checkpoint; "" skips weight loading.
        overrides:   dict merged onto the yaml `config` (defaults to the
                     requested `drivor_overrides`).
        seed:        global torch seed (`+seed=2` by default); None to skip.
        device:      "cuda" / "cpu"; defaults to cuda when available.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if overrides is None:
        overrides = DEFAULT_DRIVOR_OVERRIDES
    overrides = _absolutize_backbone_weights(dict(overrides))

    if ckpt_path and not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"DrivoR checkpoint not found at {ckpt_path}. Download it from the "
            "GitHub Releases page and place it in drivor_cem_portable/checkpoints/, "
            'or pass an explicit ckpt_path (use "" to skip weight loading).'
        )

    model = drivor_mpc.load_drivor_model(config_path, ckpt_path, overrides=overrides)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# 2) Wrap DrivoR as a CEM score_fn.
# --------------------------------------------------------------------------- #

def make_score_fn(
    drivor_model: torch.nn.Module,
    features: dict,
    subsample_indices: Optional[Iterable[int]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """(B, N, P, 3) trajectories -> (B, N) DrivoR scores (higher = better).

    `features` must contain `drivor_image` and `drivor_ego_status` (see
    `drivor_mpc.build_drivor_image` / `build_drivor_ego_status`). The scene
    backbone + ego token are computed once here and reused per call.

    Pass `subsample_indices` only when the trajectories have more poses than
    DrivoR expects (DrivoR was trained on 8 poses @ 0.5 s); leave as None /
    () for 8-pose trajectories.
    """
    return drivor_mpc.make_drivor_score_fn(
        drivor_model, features, subsample_indices=subsample_indices,
    )


# --------------------------------------------------------------------------- #
# Boundary kinematics from the DrivoR ego-status history.
# --------------------------------------------------------------------------- #

# DrivoR ego_status layout: [x, y, heading, vx, vy, ax, ay, cmd0..cmd3] (11).
_SPEED_IDX = 3
_AX_IDX = 5
_HEADING_IDX = 2


def _boundary_from_ego_status(ego_status_raw: torch.Tensor, dt: float):
    """init_speed, past_a, past_omega from a (B, T_hist, 11) ego-status tensor."""
    init_speed = ego_status_raw[:, -1, _SPEED_IDX]
    past_a, past_omega = cem_utils.past_boundary_from_history(
        ego_status_raw, dt, ax_idx=_AX_IDX, heading_idx=_HEADING_IDX,
    )
    return init_speed, past_a, past_omega


# --------------------------------------------------------------------------- #
# 3) Run CEM with DrivoR scorer + comfort + anchor costs.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def cem_refine_with_drivor(
    drivor_model: torch.nn.Module,
    features: dict,
    proposals: torch.Tensor,
    best_traj: torch.Tensor,
    *,
    subsample_indices: Optional[Iterable[int]] = None,
    init_speed: Optional[torch.Tensor] = None,
    past_a: Optional[torch.Tensor] = None,
    past_omega: Optional[torch.Tensor] = None,
    dt: float = DEFAULT_CEM_DT,
    num_iterations: int = DEFAULT_CEM_ITERATIONS,
    num_samples: int = DEFAULT_CEM_SAMPLES,
    num_elites: Optional[int] = None,
    comfort_weight: float = 0.05,
    anchor_weight: float = 0.5,
    vote: bool = True,
) -> dict:
    """CEM-refine `best_traj` using the DrivoR scorer as the CEM objective.

    The CEM objective is DrivoR-score + closed-form comfort cost + anchor
    (proximity-to-`best_traj`) regularizer — the unablated baseline of
    `cem_utils.cem_optimize`.

    Args:
        drivor_model: model from `load_drivor`.
        features:     dict with `drivor_image` and `drivor_ego_status`.
        proposals:    (B, N_p, P, 3) seed trajectories used to init the
                      Gaussian std (e.g. the model's proposal set).
        best_traj:    (B, P, 3) anchor / argmax trajectory.
        subsample_indices: pose subsampling for >8-pose trajectories.
        init_speed / past_a / past_omega: (B,) boundary kinematics; derived
            from `features["drivor_ego_status"]` when omitted.
        num_elites:   defaults to `num_samples // 8`.
        vote:         when True (default) keep the CEM trajectory only where it
            beats `best_traj` on the combined score; otherwise always return
            the CEM trajectory.

    Returns a dict with:
        trajectory      (B, P, 3) — final picked trajectory.
        cem_trajectory  (B, P, 3) — raw CEM converged mean.
        cem_score       (B,)      — combined score of the CEM trajectory.
        use_cem         (B,) bool — whether CEM was chosen (all True if not voting).
    """
    if num_elites is None:
        num_elites = max(num_samples // 8, 1)

    score_fn = make_score_fn(drivor_model, features, subsample_indices)

    device = best_traj.device
    if init_speed is None or past_a is None or past_omega is None:
        ego_status_raw = features["drivor_ego_status"].to(device=device, dtype=best_traj.dtype)
        d_speed, d_a, d_omega = _boundary_from_ego_status(ego_status_raw, dt)
        init_speed = d_speed if init_speed is None else init_speed
        past_a = d_a if past_a is None else past_a
        past_omega = d_omega if past_omega is None else past_omega

    B = best_traj.shape[0]

    cem_traj, cem_score, _ = cem_utils.cem_optimize(
        score_fn=score_fn,
        proposals=proposals,
        init_speed=init_speed,
        past_a=past_a,
        past_omega=past_omega,
        init_traj=best_traj,
        dt=dt,
        num_iterations=num_iterations,
        num_samples=num_samples,
        num_elites=num_elites,
        comfort_weight=comfort_weight,
        anchor_weight=anchor_weight,
        # Baseline (un-ablated) cost composition: scorer + comfort + anchor.
        use_scorer_cost=True,
        use_comfort_cost=True,
        use_anchor_cost=True,
    )

    out = {"cem_trajectory": cem_traj, "cem_score": cem_score}

    if not vote:
        out["trajectory"] = cem_traj
        out["use_cem"] = torch.ones(B, dtype=torch.bool, device=cem_traj.device)
        return out

    # Score the anchor under the same scorer + comfort objective and keep
    # whichever wins per-sample.
    anchor_score = score_fn(best_traj.unsqueeze(1)).squeeze(1)
    anchor_comfort = cem_utils.comfort_cost_from_traj(
        best_traj.unsqueeze(1), init_speed, past_a, past_omega, dt,
    ).squeeze(1)
    anchor_full = anchor_score - comfort_weight * anchor_comfort

    use_cem = cem_score > anchor_full
    trajectory = torch.where(use_cem[:, None, None], cem_traj, best_traj)
    out["trajectory"] = trajectory
    out["use_cem"] = use_cem
    out["anchor_score"] = anchor_full
    return out


__all__ = [
    "load_drivor",
    "make_score_fn",
    "cem_refine_with_drivor",
    "DEFAULT_DRIVOR_CONFIG_PATH",
    "DEFAULT_DRIVOR_CKPT_PATH",
    "DEFAULT_DRIVOR_OVERRIDES",
]


# --------------------------------------------------------------------------- #
# Toy example — random inputs end-to-end through DrivoR + CEM.
# --------------------------------------------------------------------------- #

def _toy_features(drivor_model, batch_size, config_path):
    """Build random DrivoR-shaped inputs (no real sensor data needed).

    Shapes mirror `drivor_mpc.build_drivor_image` / `build_drivor_ego_status`:
      drivor_image      (B, num_cams, 3, H, W)   — ImageNet-normalized cams
      drivor_ego_status (B, T_hist, 11)          — [pose(3), vel(2), acc(2), cmd(4)]
    """
    cfg = drivor_mpc.load_drivor_cfg(config_path, overrides=DEFAULT_DRIVOR_OVERRIDES)
    W, H = int(cfg["image_size"][0]), int(cfg["image_size"][1])
    num_cams = int(drivor_model.num_cams)
    device = next(drivor_model.parameters()).device

    T_hist = 4
    image = torch.randn(batch_size, num_cams, 3, H, W, device=device)
    ego_status = torch.zeros(batch_size, T_hist, 11, device=device)
    ego_status[:, :, 3] = 5.0  # vx ~ 5 m/s forward so the rollout is non-trivial
    return {"drivor_image": image, "drivor_ego_status": ego_status}, num_cams, (H, W)


def _toy_trajectories(batch_size, num_proposals, num_poses, device):
    """A fan of straight-ish proposals plus a 'best' anchor (B, P, 3)."""
    # Constant ~2.5 m/step forward, small random lateral/heading jitter.
    t = torch.arange(1, num_poses + 1, device=device, dtype=torch.float32)
    base = torch.zeros(batch_size, num_proposals, num_poses, 3, device=device)
    base[..., 0] = 2.5 * t                              # x forward
    base[..., 1] = torch.randn(batch_size, num_proposals, 1, device=device) * 0.5 * t / num_poses
    proposals = base + torch.randn_like(base) * 0.1
    best_traj = proposals[:, 0].clone()                 # pick proposal 0 as the anchor
    return proposals, best_traj


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="DrivoR + CEM toy example")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--cem-iterations", type=int, default=DEFAULT_CEM_ITERATIONS)
    parser.add_argument("--cem-samples", type=int, default=DEFAULT_CEM_SAMPLES)
    parser.add_argument("--config-path", type=str, default=DEFAULT_DRIVOR_CONFIG_PATH)
    parser.add_argument("--ckpt-path", type=str, default=DEFAULT_DRIVOR_CKPT_PATH)
    args = parser.parse_args()

    print(f"[toy] root            = {_root()}")
    print(f"[toy] config_path     = {args.config_path}")
    print(f"[toy] ckpt_path       = {args.ckpt_path}")
    print(f"[toy] ckpt exists     = {os.path.exists(args.ckpt_path)}")

    print("[toy] loading DrivoR ...")
    model = load_drivor(config_path=args.config_path, ckpt_path=args.ckpt_path)
    device = next(model.parameters()).device
    print(f"[toy] device={device}  num_cams={model.num_cams}  poses_num={model.poses_num}")

    features, num_cams, (H, W) = _toy_features(model, args.batch_size, args.config_path)
    print(f"[toy] drivor_image      {tuple(features['drivor_image'].shape)}  (cams={num_cams}, H={H}, W={W})")
    print(f"[toy] drivor_ego_status {tuple(features['drivor_ego_status'].shape)}")

    proposals, best_traj = _toy_trajectories(
        args.batch_size, args.num_proposals, model.poses_num, device,
    )
    print(f"[toy] proposals         {tuple(proposals.shape)}")
    print(f"[toy] best_traj (anchor){tuple(best_traj.shape)}")

    print("[toy] running CEM refinement with DrivoR scorer + comfort + anchor ...")
    out = cem_refine_with_drivor(
        model, features,
        proposals=proposals,
        best_traj=best_traj,
        num_iterations=args.cem_iterations,
        num_samples=args.cem_samples,
    )

    print("\n[toy] ===== results =====")
    print(f"[toy] trajectory      {tuple(out['trajectory'].shape)}")
    print(f"[toy] cem_trajectory  {tuple(out['cem_trajectory'].shape)}")
    print(f"[toy] cem_score       {out['cem_score'].tolist()}")
    print(f"[toy] anchor_score    {out['anchor_score'].tolist()}")
    print(f"[toy] use_cem         {out['use_cem'].tolist()}")
    delta = (out["trajectory"] - best_traj).norm(dim=-1).mean(dim=-1)
    print(f"[toy] mean |Δ| anchor->final per sample: {delta.tolist()}")
    print("[toy] done.")


if __name__ == "__main__":
    _main()
