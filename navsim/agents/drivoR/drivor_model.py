from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

# Comfort thresholds — mirror navsim/.../pdm_comfort_metrics.py.
_MAX_ABS_MAG_JERK = 8.37     # [m/s^3]
_MAX_ABS_LAT_ACCEL = 4.89    # [m/s^2]
_MAX_LON_ACCEL = 2.40        # [m/s^2]
_MIN_LON_ACCEL = -4.05       # [m/s^2]
_MAX_ABS_YAW_ACCEL = 1.93    # [rad/s^2]
_MAX_ABS_LON_JERK = 4.13     # [m/s^3]
_MAX_ABS_YAW_RATE = 0.95     # [rad/s]


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)

        self.b2d=config.b2d




    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Raw ego status (B, T_hist, 11) — needed by CEM for initial speed and
        # the past-boundary kinematics used by history_comfort.
        ego_status_raw: torch.Tensor = features["ego_status"]

        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = ego_status_raw.flatten(-2)
        else:
            ego_status: torch.Tensor = ego_status_raw[:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]                          # proposer output (B, N_proposer, T, 3)

        # === scorer-diversity augmentation (training only) ===
        # Append a family-diverse random bouquet of feasible trajectories to the
        # set the scorer is oracle-supervised on. This only widens the scorer's
        # supervision off the proposer's manifold; it cannot flow any gradient
        # into the proposer, since the scorer scores proposals.detach() and the
        # IL regression loops over proposal_list (proposer-only, no bouquet).
        n_aug = int(getattr(self._config, "proposal_augment_random", 0))
        if n_aug > 0:
            aug = self._random_feasible_bouquet(
                ego_status_raw[:, -1, 3], n_aug
            ).to(proposals.dtype)
            proposals = torch.cat([proposals, aug], dim=1)   # scorer set = N_proposer + n_aug
        # =====================================================

        output={}
        output["proposals"] = proposals                      # full set (proposer + aug) -> scorer + oracle
        output["proposal_list"] = proposal_list              # proposer ONLY -> IL regression target (UNCHANGED)

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        # print(
        #     self._config.noc,
        #     self._config.dac,
        #     self._config.ddc,
        #     self._config.ttc,
        #     self._config.ep,
        #     self._config.comfort,
        # )

        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +    
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()  
        + self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]
        argmax_score = pdm_score[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        # Cross-Entropy Method refinement at inference time.
        if (not self.training) and getattr(self._config, "use_cem", True):
            cem_traj, cem_score = self.cem_refine(
                proposals, scene_features, ego_token, ego_status_raw,
                init_traj=trajectory,
            )
            # Symmetric scoring: rate the argmax pick with the same combined
            # metric (pdm - lambda * comfort) that CEM is optimizing. Without
            # this, CEM is unfairly penalized by comfort and can only "win" on
            # raw pdm — which encourages scorer-exploitation drift.
            cem_dt = getattr(self._config, "cem_dt", 0.5)
            comfort_w = getattr(self._config, "cem_comfort_weight", 0.05)
            argmax_comfort = self._comfort_cost_from_traj(
                trajectory.unsqueeze(1), ego_status_raw, cem_dt
            ).squeeze(1)
            argmax_full_score = argmax_score - comfort_w * argmax_comfort

            use_cem = cem_score > argmax_full_score
            trajectory = torch.where(use_cem[:, None, None], cem_traj, trajectory)
            output["trajectory"] = trajectory
            output["cem_trajectory"] = cem_traj
            output["cem_score"] = cem_score
            output["argmax_score"] = argmax_score
            output["argmax_full_score"] = argmax_full_score
            output["use_cem"] = use_cem

        return output

    @staticmethod
    def _load_cem_anchors(anchor_path: str, num_poses: int) -> torch.Tensor:
        """Load kmeans (x, y) anchor trajectories and derive headings.

        The anchor file stores (N, P, 2) positions in the local ego frame.
        Heading at each pose is taken as the direction of the segment leading
        into it (origin prepended), matching the bicycle-rollout convention
        used by `_proposals_to_controls`.

        Returns:
            anchors: (1, N, P, 3) — (x, y, heading), batch dim for expand().
        """
        xy = torch.from_numpy(np.load(anchor_path)).float()  # (N, P, 2)
        assert xy.dim() == 3 and xy.shape[-1] == 2, f"unexpected anchor shape {tuple(xy.shape)}"
        assert xy.shape[1] == num_poses, (
            f"anchor poses ({xy.shape[1]}) != model num_poses ({num_poses})"
        )
        origin = torch.zeros(xy.shape[0], 1, 2)
        seg = torch.diff(torch.cat([origin, xy], dim=1), dim=1)  # (N, P, 2)
        heading = torch.atan2(seg[..., 1], seg[..., 0])          # (N, P)
        return torch.cat([xy, heading.unsqueeze(-1)], dim=-1).unsqueeze(0)

    def _score_trajectories(
        self,
        trajectories: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of trajectories with the same pipeline used in `forward`.

        Args:
            trajectories: (B, N, num_poses, state_size)
            scene_features: (B, S, d_model)
            ego_token: (B, 1, d_model)

        Returns:
            pdm_score: (B, N) — higher is better.
        """
        B, N = trajectories.shape[:2]
        embedded = self.pos_embed(trajectories.reshape(B, N, -1))
        tr_out = self.scorer_attention(embedded, scene_features)
        tr_out = tr_out + ego_token
        pred_logit, _, _, _, _, _, _ = self.scorer(trajectories, tr_out)

        pdm_score = (
            self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
            self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
            self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +
            (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
             self._config.ep * pred_logit['ego_progress'].sigmoid() +
             self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )
        return pdm_score

    @torch.no_grad()
    def cem_refine(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        ego_status_raw: torch.Tensor,
        init_traj: torch.Tensor = None,
    ):
        """Cross-Entropy Method in (lon-accel, yaw-rate) control space.

        Trajectories are produced by a kinematic-bicycle rollout, so all
        candidates are smooth by construction. The cost combines:
            * scorer pdm (collisions / drivable / ttc / progress / comfort head)
              evaluated on the rolled-out trajectory,
            * a closed-form comfort cost over (past history + future), which
              captures both `comfort` and `history_comfort`,
            * an anchor cost that pulls toward the argmax-proposal's controls
              to prevent CEM from drifting into out-of-distribution regions
              where the scorer over-estimates pdm.

        The final trajectory returned is the **converged Gaussian mean** —
        not the highest-scoring random sample. The mean is a smooth function
        of the input scene, which keeps consecutive frames consistent and
        protects `two_frame_extended_comfort`.
        """
        cfg = self._config
        dt = getattr(cfg, "cem_dt", 0.5)
        num_iterations = getattr(cfg, "cem_num_iterations", 5)
        num_samples = getattr(cfg, "cem_num_samples", proposals.shape[1])
        num_elites = getattr(cfg, "cem_num_elites", None)
        # num_elites = max(num_samples // 8, 1) #min(num_elites, max(num_samples // 8, 1)) 
        if num_elites is None:
            num_elites = max(num_samples // 8, 1)
        
        alpha = getattr(cfg, "cem_alpha", 0.7)
        std_floor_a = getattr(cfg, "cem_std_floor_a", 0.1)           # [m/s^2]
        std_floor_omega = getattr(cfg, "cem_std_floor_omega", 0.025) # [rad/s]
        comfort_weight = getattr(cfg, "cem_comfort_weight", 0.05)
        anchor_weight = getattr(cfg, "cem_anchor_weight", 0.5)
        pos_anchor_weight = getattr(cfg, "cem_pos_anchor_weight", 0.5)  # [/m^2]
        init_std_scale = getattr(cfg, "cem_init_std_scale", 0.5)
        rng_seed = getattr(cfg, "cem_seed", 0)

        B = proposals.shape[0]
        P = self.poses_num
        device = proposals.device
        dtype = proposals.dtype

        # Local generator → CEM is a deterministic function of the inputs.
        # Frees `two_frame_extended_comfort` from any global-RNG noise.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(rng_seed))

        init_speed = ego_status_raw[:, -1, 3]
        past_a, past_omega = self._past_boundary(ego_status_raw, dt)

        # Inverse-kinematics seeding from proposals.
        controls_init = self._proposals_to_controls(proposals, init_speed, dt)  # (B, N_p, P, 2)
        flat_init = controls_init.reshape(B, controls_init.shape[1], P * 2)

        # Anchor: argmax-proposal's controls. CEM is an L2-regularized refinement of this.
        if init_traj is not None:
            anchor_ctrl = self._proposals_to_controls(init_traj.unsqueeze(1), init_speed, dt)
            anchor_flat = anchor_ctrl.reshape(B, P * 2)
            anchor_xy = init_traj[..., :2]                  # (B, P, 2)
            anchor_x_final = init_traj[:, -1, 0]            # (B,)
        else:
            anchor_flat = flat_init.mean(dim=1)
            anchor_xy = proposals.mean(dim=1)[..., :2]
            anchor_x_final = proposals.mean(dim=1)[:, -1, 0]

        mean = anchor_flat.clone()
        std = flat_init.std(dim=1, unbiased=False) * init_std_scale
        floor = torch.tensor(
            [std_floor_a, std_floor_omega], device=device, dtype=dtype
        ).repeat(P)
        std = torch.maximum(std, floor.unsqueeze(0))

        for _ in range(num_iterations):
            eps = torch.randn(
                B, num_samples, P * 2,
                device=device, dtype=dtype, generator=gen,
            )
            sampled_flat = mean.unsqueeze(1) + eps * std.unsqueeze(1)
            controls = self._clamp_controls(sampled_flat.reshape(B, num_samples, P, 2))

            traj, speeds = self._bicycle_rollout(controls, init_speed, dt)
            scorer_score = self._score_trajectories(traj, scene_features, ego_token)
            comfort_cost = self._direct_comfort_cost(
                controls, speeds, past_a, past_omega, dt
            )

            # (a, omega) anchor — caps control deviation.
            ctrl_flat = controls.reshape(B, num_samples, P * 2)
            ctrl_anchor_cost = ((ctrl_flat - anchor_flat.unsqueeze(1)) ** 2).mean(-1)

            # (x, y) anchor — caps the *integrated* lateral/longitudinal drift,
            # which is what `lane_keeping` actually penalizes.
            xy_anchor_cost = (
                (traj[..., :2] - anchor_xy.unsqueeze(1)) ** 2
            ).sum(-1).mean(-1)



            score = (
                scorer_score
                - comfort_weight * comfort_cost
                - anchor_weight * ctrl_anchor_cost
                - pos_anchor_weight * xy_anchor_cost

            )

            # Refit Gaussian to elites (in clamped-control space).
            _, elite_idx = torch.topk(score, num_elites, dim=1)
            elite = torch.gather(
                ctrl_flat, 1, elite_idx.unsqueeze(-1).expand(-1, -1, P * 2)
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

        # Final output: converged Gaussian mean — deterministic given inputs.
        ctrl_mean = self._clamp_controls(mean.reshape(B, 1, P, 2))
        traj_mean, sp_mean = self._bicycle_rollout(ctrl_mean, init_speed, dt)
        sc_mean = self._score_trajectories(traj_mean, scene_features, ego_token).squeeze(1)
        cf_mean = self._direct_comfort_cost(
            ctrl_mean, sp_mean, past_a, past_omega, dt
        ).squeeze(1)
        final_score = sc_mean - comfort_weight * cf_mean
        return traj_mean.squeeze(1), final_score

    @torch.no_grad()
    def _comfort_cost_from_traj(
        self,
        trajectory: torch.Tensor,
        ego_status_raw: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Comfort cost for an arbitrary (B, N, P, 3) trajectory. Used to score
        the argmax candidate under the same metric as the CEM candidate."""
        init_speed = ego_status_raw[:, -1, 3]
        past_a, past_omega = self._past_boundary(ego_status_raw, dt)
        controls = self._proposals_to_controls(trajectory, init_speed, dt)
        _, speeds = self._bicycle_rollout(controls, init_speed, dt)
        return self._direct_comfort_cost(controls, speeds, past_a, past_omega, dt)

    def _random_feasible_bouquet(self, init_speed: torch.Tensor, num_proposals: int) -> torch.Tensor:
        """Family-diverse random bouquet of feasible trajectories (training aug).

        Samples (accel, yaw-rate) control sequences in 3 families and rolls them
        out with the kinematic bicycle from the current ego speed. A diverse mix
        (not constant-arc only) is what buys search-robustness — a constant-only
        bouquet still collapses under deep search.

        Args:
            init_speed: (B,) current ego longitudinal speed.
            num_proposals: number of bouquet trajectories N to generate.

        Returns:
            trajectories: (B, N, T, 3) in the local ego frame at t=0.
        """
        B = init_speed.shape[0]
        N = num_proposals
        T = self.poses_num
        dev, dtp = init_speed.device, init_speed.dtype
        dt = 0.5
        a_min, a_max, om_max = -4.0, 2.4, 0.5

        def _fam(n, fam):
            if fam == "smooth":                       # linearly-ramping controls a0->a1, o0->o1 (clothoid-like)
                t = torch.linspace(0.0, 1.0, T, device=dev, dtype=dtp).view(1, 1, T)
                a0 = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(a_min, a_max)
                a1 = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(a_min, a_max)
                o0 = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(-om_max, om_max)
                o1 = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(-om_max, om_max)
                a, om = a0 + (a1 - a0) * t, o0 + (o1 - o0) * t
            elif fam == "freeform":                   # per-step Gaussian around a trajectory-level mean (~ CEM sampling law)
                a0 = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(a_min, a_max)
                o0 = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(-om_max, om_max)
                a = (a0 + torch.randn(B, n, T, device=dev, dtype=dtp) * 0.25 * (a_max - a_min)).clamp(a_min, a_max)
                om = (o0 + torch.randn(B, n, T, device=dev, dtype=dtp) * 0.5 * om_max).clamp(-om_max, om_max)
            else:                                      # constant: one (a, omega) held for the whole horizon
                a = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(a_min, a_max).expand(B, n, T)
                om = torch.empty(B, n, 1, device=dev, dtype=dtp).uniform_(-om_max, om_max).expand(B, n, T)
            return torch.stack([a, om], dim=-1)        # (B, n, T, 2)

        mode = getattr(self._config, "random_bouquet_control_mode", "family_mix")
        if mode in ("constant", "smooth", "freeform"):
            controls = _fam(N, mode)
        else:                                          # family_mix: 1/3 each (remainder -> constant)
            n3 = N // 3
            controls = torch.cat(
                [_fam(N - 2 * n3, "constant"), _fam(n3, "smooth"), _fam(n3, "freeform")], dim=1
            )
        traj, _ = self._bicycle_rollout(controls, init_speed, dt)   # (B, N, T, 3)
        return traj

    @staticmethod
    def _bicycle_rollout(
        controls: torch.Tensor,
        init_speed: torch.Tensor,
        dt: float,
    ):
        """Forward-Euler kinematic-bicycle rollout with mid-point evaluation.

        Args:
            controls: (B, N, P, 2) — per-step (a, omega) in (m/s^2, rad/s).
            init_speed: (B,) — initial longitudinal speed.
            dt: time step in seconds.

        Returns:
            trajectory: (B, N, P, 3) — (x, y, heading) at each future step
                in the local ego frame at t=0.
            speeds_end: (B, N, P) — speed at the end of each step.
        """
        B, N, P, _ = controls.shape
        a = controls[..., 0]
        omega = controls[..., 1]

        # Speed integration (clamp to forward motion).
        v0 = init_speed.view(B, 1).expand(B, N).contiguous()
        delta_v = a * dt
        speeds_end = (v0[..., None] + torch.cumsum(delta_v, dim=-1)).clamp_min(0.0)
        speeds_start = torch.cat([v0[..., None], speeds_end[..., :-1]], dim=-1)
        v_mid = 0.5 * (speeds_start + speeds_end)

        # Heading integration.
        delta_theta = omega * dt
        headings_end = torch.cumsum(delta_theta, dim=-1)
        zero = torch.zeros(B, N, 1, device=controls.device, dtype=controls.dtype)
        headings_start = torch.cat([zero, headings_end[..., :-1]], dim=-1)
        theta_mid = 0.5 * (headings_start + headings_end)

        # Position integration via mid-point speed/heading.
        dx = v_mid * torch.cos(theta_mid) * dt
        dy = v_mid * torch.sin(theta_mid) * dt
        x = torch.cumsum(dx, dim=-1)
        y = torch.cumsum(dy, dim=-1)

        trajectory = torch.stack([x, y, headings_end], dim=-1)
        return trajectory, speeds_end

    @staticmethod
    def _proposals_to_controls(
        proposals: torch.Tensor,
        init_speed: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Inverse kinematics: (B, N, P, 3) trajectories -> (B, N, P, 2) (a, omega).

        Used to seed the CEM Gaussian from the model's proposals.
        """
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
        dh = _wrap_angle(torch.diff(h_full, dim=-1))

        # Per-step speed (proxy via segment length / dt).
        speed = torch.sqrt(dx * dx + dy * dy) / dt
        omega = dh / dt

        # Acceleration via finite-diff of speed (with init_speed as the prior).
        v0 = init_speed.view(B, 1, 1).expand(B, N, 1)
        speed_full = torch.cat([v0, speed], dim=-1)
        a = torch.diff(speed_full, dim=-1) / dt

        return torch.stack([a, omega], dim=-1)

    @staticmethod
    def _past_boundary(ego_status_raw: torch.Tensor, dt: float):
        """Last past (lon-accel, yaw-rate). Used as the t=-dt sample to
        compute boundary jerk and yaw-acceleration (i.e. history comfort)."""
        B = ego_status_raw.shape[0]
        device = ego_status_raw.device
        dtype = ego_status_raw.dtype
        if ego_status_raw.dim() != 3 or ego_status_raw.shape[1] < 2:
            zero = torch.zeros(B, device=device, dtype=dtype)
            return zero, zero
        # ego_status layout: [x, y, heading, vx, vy, ax, ay, cmd0..cmd3]
        last_ax = ego_status_raw[:, -1, 5]
        h_prev = ego_status_raw[:, -2, 2]
        h_curr = ego_status_raw[:, -1, 2]
        last_omega = _wrap_angle(h_curr - h_prev) / dt
        return last_ax, last_omega

    @staticmethod
    def _clamp_controls(controls: torch.Tensor) -> torch.Tensor:
        """Clamp (a, omega) to slightly-relaxed kinematic envelopes. Keeps the
        rollout in a sane regime even when the Gaussian samples extreme values.
        """
        margin = 1.5
        a = controls[..., 0].clamp(_MIN_LON_ACCEL * margin, _MAX_LON_ACCEL * margin)
        omega = controls[..., 1].clamp(
            -_MAX_ABS_YAW_RATE * margin, _MAX_ABS_YAW_RATE * margin
        )
        return torch.stack([a, omega], dim=-1)

    @staticmethod
    def _direct_comfort_cost(
        controls: torch.Tensor,
        speeds: torch.Tensor,
        past_a: torch.Tensor,
        past_omega: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Closed-form comfort penalty (lower = better) over the joint
        (past-boundary + future) sequence.

        Penalizes any threshold violation of: longitudinal acceleration and
        jerk, lateral acceleration, magnitude jerk, yaw rate, and yaw
        acceleration. The boundary jerk / yaw_accel terms — built by
        prepending the last past (a, omega) — are exactly what makes
        `history_comfort` differ from `comfort`.
        """
        B, N, P, _ = controls.shape
        a = controls[..., 0]      # (B, N, P)
        omega = controls[..., 1]  # (B, N, P)

        # Prepend the past boundary so jerk / yaw-accel cover the history join.
        a_prev = past_a.view(B, 1, 1).expand(B, N, 1)
        om_prev = past_omega.view(B, 1, 1).expand(B, N, 1)
        a_full = torch.cat([a_prev, a], dim=-1)        # (B, N, P+1)
        om_full = torch.cat([om_prev, omega], dim=-1)
        lon_jerk = torch.diff(a_full, dim=-1) / dt     # (B, N, P)
        yaw_accel = torch.diff(om_full, dim=-1) / dt

        # Lateral accel ~ v * omega (centripetal in body frame).
        lat_accel = speeds * omega

        # Magnitude jerk: combine lon and lat accel changes.
        zero = torch.zeros(B, N, 1, device=controls.device, dtype=controls.dtype)
        lat_full = torch.cat([zero, lat_accel], dim=-1)
        d_lat = torch.diff(lat_full, dim=-1) / dt
        mag_jerk = torch.sqrt(lon_jerk * lon_jerk + d_lat * d_lat + 1e-8)

        cost = (
            F.relu(a - _MAX_LON_ACCEL).pow(2).sum(-1)
            + F.relu(_MIN_LON_ACCEL - a).pow(2).sum(-1)
            + F.relu(omega.abs() - _MAX_ABS_YAW_RATE).pow(2).sum(-1)
            + F.relu(lon_jerk.abs() - _MAX_ABS_LON_JERK).pow(2).sum(-1)
            + F.relu(yaw_accel.abs() - _MAX_ABS_YAW_ACCEL).pow(2).sum(-1)
            + F.relu(lat_accel.abs() - _MAX_ABS_LAT_ACCEL).pow(2).sum(-1)
            + F.relu(mag_jerk - _MAX_ABS_MAG_JERK).pow(2).sum(-1)
        )
        return cost



