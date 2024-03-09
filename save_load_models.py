from agent_1 import Actor
from agent_1 import Critic_Coordinator as Critic
from agent_1 import Psi
import torch
import numpy as np
import re
import pathlib
import pickle
import os
from dotmap import DotMap

_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")

def start_or_resume_from_checkpoint(data):
    """
    Create actor, critic, actor optimizer and critic optimizer from scratch
    or load from latest checkpoint if it exists.
    """
    BASE_CHECKPOINT_PATH = data['Checkpoint_path']
    stop_conditions = data['stop_conditions']
    actor = data['Actor']
    critic = data['Critic']
    obsv_dim = data['State_dim']
    action_dim = data['Action_dim']
    continuous_action_space = data['continuous_action_space']
    actor_optimizer = data['Actor_optimizer']
    critic_optimizer = data['Critic_optimizer']
    hp = data['hp']
    TRAIN_DEVICE = data['device']

    psi = data['psi']
    psi_optimizer = data['psi_optimizer']

    max_checkpoint_iteration = get_last_checkpoint_iteration(BASE_CHECKPOINT_PATH)

    # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
    if max_checkpoint_iteration > 0:
        actor_params, critic_params, psi_params, stop_conditions = \
            load_checkpoint(max_checkpoint_iteration, BASE_CHECKPOINT_PATH, TRAIN_DEVICE, hp)

        actor.load_state_dict(actor_params['actor_parameters'], strict=True)
        critic.load_state_dict(critic_params['critic_parameters'], strict=True)
        psi.load_state_dict(psi_params['psi_parameters'], strict=True)



        actor_optimizer.load_state_dict(actor_params['optimiser_params'])
        critic_optimizer.load_state_dict(critic_params['optimiser_params'])
        psi_optimizer.load_state_dict(psi_params['optimiser_params'])

        '''We have to move manually move optimizer states to 
        TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.#'''

        for state in actor_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

        for state in critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

        for state in psi_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

    return actor, critic, psi, actor_optimizer, critic_optimizer, psi_optimizer, max_checkpoint_iteration, stop_conditions


def get_last_checkpoint_iteration(BASE_CHECKPOINT_PATH):
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(BASE_CHECKPOINT_PATH):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(BASE_CHECKPOINT_PATH)])
        print('Loading model from last checkpoint at:', BASE_CHECKPOINT_PATH, "\nIteration: ", max_checkpoint_iteration)
    else:
        max_checkpoint_iteration = 0
        print('No previous checkpoint, starting fresh!')
    return max_checkpoint_iteration


def load_checkpoint(iteration, BASE_CHECKPOINT_PATH, TRAIN_DEVICE, hp):
    """
    Load from training checkpoint.
    """
    CHECKPOINT_PATH = os.path.join(BASE_CHECKPOINT_PATH, str(iteration))
    with open(CHECKPOINT_PATH + "/parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)

    assert hp.ENV == checkpoint.env, "To resume training environment must match current settings."
    # assert ENV_MASK_VELOCITY == checkpoint.env_mask_velocity, \
    #     "To resume training model architecture must match current settings."
    assert list(hp.__dict__.values()) == list(checkpoint.hp.__dict__.values()), \
        "To resume training hyper-parameters must match current settings."

    actor_state_dict = torch.load(CHECKPOINT_PATH + "/actor.pt", map_location=torch.device(TRAIN_DEVICE))
    critic_state_dict = torch.load(CHECKPOINT_PATH + "/critic.pt", map_location=torch.device(TRAIN_DEVICE))

    psi_state_dict = torch.load(CHECKPOINT_PATH + "/psi.pt", map_location=torch.device(TRAIN_DEVICE))

    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "/actor_optimizer.pt",
                                            map_location=torch.device(TRAIN_DEVICE))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "/critic_optimizer.pt",
                                             map_location=torch.device(TRAIN_DEVICE))
    psi_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "/psi_optimizer.pt",
                                             map_location=torch.device(TRAIN_DEVICE))
    actor_params = {'actor_parameters': actor_state_dict,
                    'optimiser_params': actor_optimizer_state_dict}
    critic_params = {'critic_parameters': critic_state_dict,
                     'optimiser_params': critic_optimizer_state_dict}
    psi_params = {'psi_parameters': psi_state_dict,
                  'optimiser_params': psi_optimizer_state_dict}

    return actor_params, critic_params, psi_params, checkpoint.stop_conditions


def save_checkpoint(actor, critic, psi, actor_optimizer, critic_optimizer, psi_optimizer,
                    iteration, stop_conditions, hp, ENV_MASK_VELOCITY,
                    BASE_CHECKPOINT_PATH):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.env = hp.ENV
    checkpoint.env_mask_velocity = ENV_MASK_VELOCITY
    checkpoint.iteration = iteration
    checkpoint.stop_conditions = stop_conditions
    checkpoint.hp = hp
    CHECKPOINT_PATH = os.path.join(BASE_CHECKPOINT_PATH, str(iteration))
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH + "/parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    with open(CHECKPOINT_PATH + "/actor_class.pt", "wb") as f:
        pickle.dump(Actor, f)
    with open(CHECKPOINT_PATH + "/critic_class.pt", "wb") as f:
        pickle.dump(Critic, f)
    with open(CHECKPOINT_PATH + "/psi_class.pt", "wb") as f:
        pickle.dump(Psi, f)

    torch.save(actor.state_dict(), CHECKPOINT_PATH + "/actor.pt")
    torch.save(critic.state_dict(), CHECKPOINT_PATH + "/critic.pt")
    torch.save(psi.state_dict(), CHECKPOINT_PATH + "/psi.pt")
    torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "/actor_optimizer.pt")
    torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "/critic_optimizer.pt")
    torch.save(psi_optimizer.state_dict(), CHECKPOINT_PATH + "/psi_optimizer.pt")


def save_parameters(writer, tag, model, batch_idx):
    """
    Save model parameters for tensorboard.
    """
    for k, v in model.state_dict().items():
        shape = v.shape
        # Fix shape definition for tensorboard.
        shape_formatted = _INVALID_TAG_CHARACTERS.sub("_", str(shape))
        # Don't do this for single weights or biases
        if np.any(np.array(shape) > 1):
            mean = torch.mean(v)
            std_dev = torch.std(v)
            maximum = torch.max(v)
            minimum = torch.min(v)
            writer.add_scalars(
                "{}_weights/{}{}".format(tag, k, shape_formatted),
                {"mean": mean, "std_dev": std_dev, "max": maximum, "min": minimum},
                batch_idx)
        else:
            writer.add_scalar("{}_{}{}".format(tag, k, shape_formatted), v.data, batch_idx)
