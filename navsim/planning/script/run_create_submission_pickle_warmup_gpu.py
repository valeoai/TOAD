import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict
from datetime import datetime
import pytorch_lightning as pl
import torch.distributed as dist
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle_gpu"


def run_test_evaluation(
    agent: AbstractAgent,
    input_loader: SceneLoader,
    model_trajectory: Dict
) -> Dict[str, Trajectory]:
    """
    Function to create the output file for evaluation of an agent on the testserver
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param synthetic_sensor_path: pathlib path to sensor blobs
    :param synthetic_scenes_path: pathlib path to synthetic scenes
    :param save_path: pathlib path to folder where scores are stored as .csv
    """
    # first stage output
    first_stage_output: Dict[str, Trajectory] = {}
    for token in tqdm(input_loader.tokens_stage_one, desc="Running first stage evaluation"):
        try:
            trajectory = model_trajectory[token]['trajectory']
            first_stage_output.update({token: trajectory})
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    # second stage output

    scene_loader_tokens_stage_two = input_loader.reactive_tokens_stage_two

    second_stage_output: Dict[str, Trajectory] = {}
    for token in tqdm(scene_loader_tokens_stage_two, desc="Running second stage evaluation"):
        try:
            trajectory = model_trajectory[token]['trajectory']
            first_stage_output.update({token: trajectory})
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    return first_stage_output, second_stage_output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), cfg.experiment_name)
    os.makedirs(dump_root, exist_ok=True)
    dump_path = os.path.join(dump_root, f"{timestamp}.pkl")


    agent = instantiate(cfg.agent)
    data_path = Path(cfg.navsim_log_path)
    synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=agent.get_sensor_config(),
    )
    agent.initialize()
    try:
        pl.seed_everything(cfg.seed, workers=True)
        print(f"!!!!!!Global Seed set to {cfg.seed}")
    except Exception as e:
        pl.seed_everything(2, workers=True)
        print(f"!!!!!!Global Seed set to 2")  

    dataset = Dataset(
        scene_loader=input_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(
        AgentLightningModule(
            agent=agent,
        ),
        dataloader,
        return_predictions=True
    )

    dist.barrier()
    all_predictions = [None for _ in range(dist.get_world_size())]

    if dist.is_initialized():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions.append(predictions)

    if dist.get_rank() != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    pickle.dump(merged_predictions, open(dump_path, 'wb'))


    first_stage_output, second_stage_output = run_test_evaluation(
        agent=agent,
        input_loader=input_loader,
        model_trajectory=merged_predictions
    )

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "first_stage_predictions": [first_stage_output],
        "second_stage_predictions": [second_stage_output],
    }

    # pickle and save dict
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission filed was saved to {filename}")


if __name__ == "__main__":
    main()
