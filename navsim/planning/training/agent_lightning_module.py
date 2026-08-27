# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, Tuple, List

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch

        prediction = self.agent.forward(features)
        loss, loss_dict = self.agent.compute_loss(features, targets, prediction, tokens)

        for k, v in loss_dict.items():
            self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        # PAD / DrivoR-style agents reach the model via agent.forward(features).
        # Stash tokens here so PadModel.cem_refine can build a GT-PDMS score_fn
        # without changing the AbstractAgent signature.
        features["_tokens"] = list(tokens)
        # Same plumbing for the rasterized-GT pseudo-PDMS scorer: when the
        # agent registered a RasterScoreTargetBuilder, its raster targets ride
        # into the model through features["_raster_targets"].

        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]

            if False: #self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                # _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for batch_idx, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if False: #self.for_viz:
                proposal_list = [proposal_list[batch_idx].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposal_list, 
                    # 'all_proposal_scores': final_scores[index],
                    'high_level_command': ego_status[batch_idx]
                }
            else:
                result[token] = {'trajectory': proposal, "pdm_score": predictions["pdm_score"].cpu().numpy()[batch_idx], "proposals": predictions["proposals"].cpu().numpy()[batch_idx]}
        return result

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        return self.predict_step_drivor(batch, batch_idx)
