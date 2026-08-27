"""Cross-Entropy Method (CEM) utilities for inference-time MPC refinement.

Mirrors the formulation used in `navsim/agents/drivoR/drivor_model.py`. The
core CEM loop is model-agnostic: it operates on (lon-accel, yaw-rate) controls,
rolls them out with a kinematic-bicycle model, and selects elites by a score
that combines the per-model scorer with a closed-form comfort cost and an
anchor regularizer. Each agent supplies its own `score_fn` callable.

Trajectories are produced by the rollout, so candidates are smooth by
construction. The final trajectory returned is the **converged Gaussian mean**
(deterministic given inputs), which keeps consecutive frames consistent and
preserves `two_frame_extended_comfort`.
"""

from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from navsim.agents.utils.inference_profiler import InferenceTimer


_PROFILER = InferenceTimer("cem", log_every=50)


# Comfort thresholds — mirror navsim/.../pdm_comfort_metrics.py.
MAX_ABS_MAG_JERK = 8.37     # [m/s^3]
MAX_ABS_LAT_ACCEL = 4.89    # [m/s^2]
MAX_LON_ACCEL = 2.40        # [m/s^2]
MIN_LON_ACCEL = -4.05       # [m/s^2]
MAX_ABS_YAW_ACCEL = 1.93    # [rad/s^2]
MAX_ABS_LON_JERK = 4.13     # [m/s^3]
MAX_ABS_YAW_RATE = 0.95     # [rad/s]

# Named comfort sub-terms used by the cost-term ablation (#2). Pass any
# subset of these as `comfort_terms=` to `direct_comfort_cost` /
# `cem_optimize` / `comfort_cost_from_traj`. `None` keeps every term enabled,
# which preserves the unmodified baseline behavior.
ALL_COMFORT_TERMS: Tuple[str, ...] = (
    "lon_accel", "yaw_rate", "lon_jerk", "yaw_accel", "lat_accel", "mag_jerk",
)


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def bicycle_rollout(
    controls: torch.Tensor,
    init_speed: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-Euler kinematic-bicycle rollout with mid-point evaluation.

    Args:
        controls: (B, N, P, 2) per-step (a, omega) in (m/s^2, rad/s).
        init_speed: (B,) initial longitudinal speed.
        dt: time step in seconds.

    Returns:
        trajectory: (B, N, P, 3) (x, y, heading) in the local ego frame at t=0.
        speeds_end: (B, N, P) speed at the end of each step.
    """
    B, N, P, _ = controls.shape
    a = controls[..., 0]
    omega = controls[..., 1]

    v0 = init_speed.view(B, 1).expand(B, N).contiguous()
    delta_v = a * dt
    speeds_end = (v0[..., None] + torch.cumsum(delta_v, dim=-1)).clamp_min(0.0)
    speeds_start = torch.cat([v0[..., None], speeds_end[..., :-1]], dim=-1)
    v_mid = 0.5 * (speeds_start + speeds_end)

    delta_theta = omega * dt
    headings_end = torch.cumsum(delta_theta, dim=-1)
    zero = torch.zeros(B, N, 1, device=controls.device, dtype=controls.dtype)
    headings_start = torch.cat([zero, headings_end[..., :-1]], dim=-1)
    theta_mid = 0.5 * (headings_start + headings_end)

    dx = v_mid * torch.cos(theta_mid) * dt
    dy = v_mid * torch.sin(theta_mid) * dt
    x = torch.cumsum(dx, dim=-1)
    y = torch.cumsum(dy, dim=-1)

    trajectory = torch.stack([x, y, headings_end], dim=-1)
    return trajectory, speeds_end


def proposals_to_controls(
    proposals: torch.Tensor,
    init_speed: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Inverse kinematics: (B, N, P, 3) trajectories -> (B, N, P, 2) (a, omega)."""
    B, N, P, _ = proposals.shape
    x = proposals[..., 0]
    y = proposals[..., 1]
    h = proposals[..., 2]

    zero = torch.zeros(B, N, 1, device=proposals.device, dtype=proposals.dtype)
    x_full = torch.cat([zero, x], dim=-1)
    y_full = torch.cat([zero, y], dim=-1)
    h_full = torch.cat([zero, h], dim=-1)

    dx = torch.diff(x_full, dim=-1)
    dy = torch.diff(y_full, dim=-1)
    dh = wrap_angle(torch.diff(h_full, dim=-1))

    speed = torch.sqrt(dx * dx + dy * dy) / dt
    omega = dh / dt

    v0 = init_speed.view(B, 1, 1).expand(B, N, 1)
    speed_full = torch.cat([v0, speed], dim=-1)
    a = torch.diff(speed_full, dim=-1) / dt

    return torch.stack([a, omega], dim=-1)


def past_boundary_from_history(
    ego_status_history: torch.Tensor,
    dt: float,
    ax_idx: int = 5,
    heading_idx: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Last past (lon-accel, yaw-rate) from a history tensor of shape (B, T, F).

    Used as the t=-dt sample so boundary jerk and yaw-acceleration cover the
    history join (i.e. `history_comfort`).
    """
    B = ego_status_history.shape[0]
    device = ego_status_history.device
    dtype = ego_status_history.dtype
    if ego_status_history.dim() != 3 or ego_status_history.shape[1] < 2:
        zero = torch.zeros(B, device=device, dtype=dtype)
        return zero, zero
    last_ax = ego_status_history[:, -1, ax_idx]
    h_prev = ego_status_history[:, -2, heading_idx]
    h_curr = ego_status_history[:, -1, heading_idx]
    last_omega = wrap_angle(h_curr - h_prev) / dt
    return last_ax, last_omega


def zero_past_boundary(
    init_speed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback boundary kinematics when no history tensor is available."""
    zero = torch.zeros_like(init_speed)
    return zero, zero


def clamp_controls(controls: torch.Tensor) -> torch.Tensor:
    """Clamp (a, omega) to slightly-relaxed kinematic envelopes."""
    margin = 1.5
    a = controls[..., 0].clamp(MIN_LON_ACCEL * margin, MAX_LON_ACCEL * margin)
    omega = controls[..., 1].clamp(
        -MAX_ABS_YAW_RATE * margin, MAX_ABS_YAW_RATE * margin
    )
    return torch.stack([a, omega], dim=-1)


def direct_comfort_cost(
    controls: torch.Tensor,
    speeds: torch.Tensor,
    past_a: torch.Tensor,
    past_omega: torch.Tensor,
    dt: float,
    terms: Optional[Iterable[str]] = None,
) -> torch.Tensor:
    """Closed-form comfort penalty (lower = better) over (past + future).

    ``terms`` selects which comfort sub-terms contribute to the cost. ``None``
    enables all six (baseline behavior). Passing a subset of ``ALL_COMFORT_TERMS``
    isolates the effect of individual constraints for ablation #2.
    """
    B, N, P, _ = controls.shape
    a = controls[..., 0]
    omega = controls[..., 1]

    a_prev = past_a.view(B, 1, 1).expand(B, N, 1)
    om_prev = past_omega.view(B, 1, 1).expand(B, N, 1)
    a_full = torch.cat([a_prev, a], dim=-1)
    om_full = torch.cat([om_prev, omega], dim=-1)
    lon_jerk = torch.diff(a_full, dim=-1) / dt
    yaw_accel = torch.diff(om_full, dim=-1) / dt

    lat_accel = speeds * omega

    zero = torch.zeros(B, N, 1, device=controls.device, dtype=controls.dtype)
    lat_full = torch.cat([zero, lat_accel], dim=-1)
    d_lat = torch.diff(lat_full, dim=-1) / dt
    mag_jerk = torch.sqrt(lon_jerk * lon_jerk + d_lat * d_lat + 1e-8)

    if terms is None:
        active = set(ALL_COMFORT_TERMS)
    else:
        active = set(terms)

    cost = torch.zeros(B, N, device=controls.device, dtype=controls.dtype)
    if "lon_accel" in active:
        cost = cost + F.relu(a - MAX_LON_ACCEL).pow(2).sum(-1)
        cost = cost + F.relu(MIN_LON_ACCEL - a).pow(2).sum(-1)
    if "yaw_rate" in active:
        cost = cost + F.relu(omega.abs() - MAX_ABS_YAW_RATE).pow(2).sum(-1)
    if "lon_jerk" in active:
        cost = cost + F.relu(lon_jerk.abs() - MAX_ABS_LON_JERK).pow(2).sum(-1)
    if "yaw_accel" in active:
        cost = cost + F.relu(yaw_accel.abs() - MAX_ABS_YAW_ACCEL).pow(2).sum(-1)
    if "lat_accel" in active:
        cost = cost + F.relu(lat_accel.abs() - MAX_ABS_LAT_ACCEL).pow(2).sum(-1)
    if "mag_jerk" in active:
        cost = cost + F.relu(mag_jerk - MAX_ABS_MAG_JERK).pow(2).sum(-1)
    return cost


def _speeds_from_traj(traj: torch.Tensor, dt: float) -> torch.Tensor:
    """End-of-step longitudinal speeds derived from finite-difference of (x, y).

    Mirrors what `bicycle_rollout` returns for `speeds_end`, but works directly
    on a trajectory sample without re-running the bicycle integrator. Used by
    the trajectory-space sampling ablation (#1+#3) so the comfort cost still
    has well-defined speeds even though we never rolled out controls.
    """
    B, N, P, _ = traj.shape
    x = traj[..., 0]
    y = traj[..., 1]
    zero = torch.zeros(B, N, 1, device=traj.device, dtype=traj.dtype)
    x_full = torch.cat([zero, x], dim=-1)
    y_full = torch.cat([zero, y], dim=-1)
    dx = torch.diff(x_full, dim=-1)
    dy = torch.diff(y_full, dim=-1)
    return torch.sqrt(dx * dx + dy * dy) / dt


@torch.no_grad()
def bicycle_smooth_traj(
    traj: torch.Tensor,
    init_speed: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Project a single (B, P, 3) trajectory through bicycle inverse + forward.

    Used for ablation #5 (CEM vs. direct bicycle smoothing): converts the
    argmax trajectory to controls via inverse kinematics, clamps to the
    physical envelope, and rolls them out again. Any change from CEM is the
    pure effect of the kinematic-bicycle projection, with no optimization.
    """
    controls = proposals_to_controls(traj.unsqueeze(1), init_speed, dt)
    controls = clamp_controls(controls)
    smoothed, _ = bicycle_rollout(controls, init_speed, dt)
    return smoothed.squeeze(1)


@torch.no_grad()
def cem_optimize(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    proposals: torch.Tensor,
    init_speed: torch.Tensor,
    past_a: torch.Tensor,
    past_omega: torch.Tensor,
    init_traj: Optional[torch.Tensor] = None,
    dt: float = 0.5,
    num_iterations: int = 5,
    num_samples: Optional[int] = None,
    num_elites: Optional[int] = None,
    alpha: float = 1.0, #0.7,
    std_floor_a: float = 0.1,
    std_floor_omega: float = 0.025,
    comfort_weight: float = 0.05,
    anchor_weight: float = 0.5,
    init_std_scale: float = 0.5,
    # ---------------- Ablation knobs ----------------
    init_mode: str = "default",          # ablation #0: "default" | "random"
    sample_space: str = "controls",      # ablations #1+#3: "controls" | "traj"
    use_scorer_cost: bool = True,        # ablation #2: drop model scorer term
    use_comfort_cost: bool = True,       # ablation #2: drop comfort cost term
    use_anchor_cost: bool = True,        # ablation #2: drop anchor-distance term
    comfort_terms: Optional[Iterable[str]] = None,  # ablation #2 sub-terms
    random_init_std_a: float = 1.0,      # wide std for random init (controls space)
    random_init_std_omega: float = 0.2,
    random_init_std_xy: float = 2.0,     # wide std for random init (traj space)
    random_init_std_heading: float = 0.2,
    traj_std_floor_xy: float = 0.1,
    traj_std_floor_heading: float = 0.01,
    # ---------------- Trace logging (visualization only) ----------------
    trace: Optional[list] = None,
    trace_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CEM and return the converged-mean trajectory.

    Args:
        score_fn: takes (B, N, P, 3) trajectories, returns (B, N) score (higher = better).
        proposals: (B, N_p, P, 3) used to seed the initial Gaussian std.
        init_speed: (B,) initial longitudinal speed.
        past_a, past_omega: (B,) last past acceleration / yaw-rate.
        init_traj: optional (B, P, 3) anchor (e.g. argmax pick).
        dt: time step seconds. Other args follow drivoR defaults.

        init_mode: "default" seeds mean/std from `proposals` + `init_traj`
            (baseline). "random" replaces them with a zero mean and wide
            constant std, so the CEM is initialized without using model
            outputs — when combined with `sample_space="controls"` this is
            "random init smoothed by bicycle" (ablation #0).
        sample_space: "controls" perturbs (a, omega) and rolls out via the
            kinematic-bicycle model (baseline). "traj" perturbs the
            trajectory tensor directly with no rollout — covers ablations
            #1 (drop bicycle dynamics) and #3 (drop controls reparameterization).
        use_scorer_cost / use_comfort_cost / use_anchor_cost: toggle each
            macro cost term independently (ablation #2 macro).
        comfort_terms: optional subset of ``ALL_COMFORT_TERMS`` to isolate
            individual comfort constraints (ablation #2 sub-terms).

    Returns:
        traj_mean: (B, P, 3) — final (converged-mean) trajectory.
        final_score: (B,) — combined score of the returned trajectory.
        comfort_cost: (B,) — closed-form comfort cost of the returned trajectory.
    """
    if init_mode not in ("default", "random"):
        raise ValueError(f"Unknown init_mode: {init_mode!r}")
    if sample_space not in ("controls", "traj"):
        raise ValueError(f"Unknown sample_space: {sample_space!r}")

    B = proposals.shape[0]
    P = proposals.shape[2]
    device = proposals.device
    dtype = proposals.dtype
    if num_samples is None:
        num_samples = proposals.shape[1]
    if num_elites is None:
        num_elites = max(num_samples // 8, 1)

    # Per-pose latent dim depends on the sample space.
    D = 2 if sample_space == "controls" else 3

    # --------- Initial Gaussian mean / std / anchor ---------
    if init_mode == "default":
        if sample_space == "controls":
            controls_init = proposals_to_controls(proposals, init_speed, dt)
            flat_init = controls_init.reshape(B, controls_init.shape[1], P * D)
            if init_traj is not None:
                anchor_ctrl = proposals_to_controls(init_traj.unsqueeze(1), init_speed, dt)
                anchor_flat = anchor_ctrl.reshape(B, P * D)
            else:
                anchor_flat = flat_init.mean(dim=1)
        else:  # traj space
            flat_init = proposals.reshape(B, proposals.shape[1], P * D)
            if init_traj is not None:
                anchor_flat = init_traj.reshape(B, P * D)
            else:
                anchor_flat = flat_init.mean(dim=1)
        mean = anchor_flat.clone()
        std = flat_init.std(dim=1, unbiased=False) * init_std_scale
    else:  # random
        # Zero anchor (constant velocity / origin) + wide constant std. Drops
        # all model-derived information from the CEM seed (ablation #0).
        anchor_flat = torch.zeros(B, P * D, device=device, dtype=dtype)
        mean = anchor_flat.clone()
        if sample_space == "controls":
            per_dim = torch.tensor(
                [random_init_std_a, random_init_std_omega], device=device, dtype=dtype,
            ).repeat(P)
        else:
            per_dim = torch.tensor(
                [random_init_std_xy, random_init_std_xy, random_init_std_heading],
                device=device, dtype=dtype,
            ).repeat(P)
        std = per_dim.unsqueeze(0).expand(B, -1).clone()

    # Std floor — depends on parameterization.
    if sample_space == "controls":
        floor = torch.tensor(
            [std_floor_a, std_floor_omega], device=device, dtype=dtype
        ).repeat(P)
    else:
        floor = torch.tensor(
            [traj_std_floor_xy, traj_std_floor_xy, traj_std_floor_heading],
            device=device, dtype=dtype,
        ).repeat(P)
    std = torch.maximum(std, floor.unsqueeze(0))

    def _decode_mean(m: torch.Tensor) -> torch.Tensor:
        """Decode the current Gaussian mean to a (B, 1, P, 3) trajectory."""
        if sample_space == "controls":
            ctrl_m = clamp_controls(m.reshape(B, 1, P, D))
            tr_m, _ = bicycle_rollout(ctrl_m, init_speed, dt)
        else:
            tr_m = m.reshape(B, 1, P, D)
        return tr_m

    def _slice(t: torch.Tensor) -> torch.Tensor:
        if trace_indices is None:
            return t
        return t.index_select(0, trace_indices.to(t.device))

    if trace is not None:
        trace.append({
            "iter": 0,
            "samples_traj": None,
            "scores": None,
            "elite_idx": None,
            "mean_traj": _slice(_decode_mean(mean)).detach().cpu(),
            "std": _slice(std).detach().cpu(),
        })

    for it in range(num_iterations):
        eps = torch.randn(B, num_samples, P * D, device=device, dtype=dtype)
        sampled_flat = mean.unsqueeze(1) + eps * std.unsqueeze(1)

        if sample_space == "controls":
            controls = clamp_controls(sampled_flat.reshape(B, num_samples, P, D))
            traj, speeds = bicycle_rollout(controls, init_speed, dt)
            elite_flat = controls.reshape(B, num_samples, P * D)
        else:
            # Sample is the trajectory itself; derive controls/speeds only
            # for the comfort-cost evaluation (no bicycle rollout in the loop).
            traj = sampled_flat.reshape(B, num_samples, P, D)
            controls = proposals_to_controls(traj, init_speed, dt)
            speeds = _speeds_from_traj(traj, dt)
            elite_flat = sampled_flat

        score = torch.zeros(B, num_samples, device=device, dtype=dtype)
        if use_scorer_cost:
            with _PROFILER.section("cem_iterations_scoring", batch_size=B):
                score = score + score_fn(traj)
        if use_comfort_cost:
            score = score - comfort_weight * direct_comfort_cost(
                controls, speeds, past_a, past_omega, dt, terms=comfort_terms,
            )
        if use_anchor_cost:
            anchor_cost = ((elite_flat - anchor_flat.unsqueeze(1)) ** 2).mean(-1)
            score = score - anchor_weight * anchor_cost

        _, elite_idx = torch.topk(score, num_elites, dim=1)
        elite = torch.gather(
            elite_flat, 1, elite_idx.unsqueeze(-1).expand(-1, -1, P * D)
        )
        new_mean = elite.mean(dim=1)
        if num_elites > 1:
            new_std = torch.maximum(
                elite.std(dim=1, unbiased=False), floor.unsqueeze(0)
            )
        else:
            new_std = std
        mean = alpha * new_mean + (1 - alpha) * mean
        std = alpha * new_std + (1 - alpha) * std

        if trace is not None:
            trace.append({
                "iter": it + 1,
                "samples_traj": _slice(traj).detach().cpu(),
                "scores": _slice(score).detach().cpu(),
                "elite_idx": _slice(elite_idx).detach().cpu(),
                "mean_traj": _slice(_decode_mean(mean)).detach().cpu(),
                "std": _slice(std).detach().cpu(),
            })

    # --------- Decode the converged mean into a trajectory ---------
    if sample_space == "controls":
        ctrl_mean = clamp_controls(mean.reshape(B, 1, P, D))
        traj_mean, sp_mean = bicycle_rollout(ctrl_mean, init_speed, dt)
    else:
        traj_mean = mean.reshape(B, 1, P, D)
        ctrl_mean = proposals_to_controls(traj_mean, init_speed, dt)
        sp_mean = _speeds_from_traj(traj_mean, dt)

    cf_mean = direct_comfort_cost(
        ctrl_mean, sp_mean, past_a, past_omega, dt, terms=comfort_terms,
    ).squeeze(1)

    final_score = torch.zeros(B, device=device, dtype=dtype)
    if use_scorer_cost:
        final_score = final_score + score_fn(traj_mean).squeeze(1)
    if use_comfort_cost:
        final_score = final_score - comfort_weight * cf_mean
    _PROFILER.increment()
    return traj_mean.squeeze(1), final_score, cf_mean


@torch.no_grad()
def comfort_cost_from_traj(
    trajectory: torch.Tensor,
    init_speed: torch.Tensor,
    past_a: torch.Tensor,
    past_omega: torch.Tensor,
    dt: float,
    terms: Optional[Iterable[str]] = None,
) -> torch.Tensor:
    """Comfort cost for an arbitrary (B, N, P, 3) trajectory."""
    controls = proposals_to_controls(trajectory, init_speed, dt)
    _, speeds = bicycle_rollout(controls, init_speed, dt)
    return direct_comfort_cost(controls, speeds, past_a, past_omega, dt, terms=terms)
