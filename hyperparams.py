import argparse
from dataclasses import dataclass


ENV_MASK_VELOCITY = False

# Default Hyperparameters
SCALE_REWARD:         float = 0.001
MIN_REWARD:           float = -1000.
HIDDEN_SIZE:          int = 128
BATCH_SIZE:           int = 512
DISCOUNT:             float = 0.99
GAE_LAMBDA:           float = 0.95
PPO_CLIP:             float = 0.2
PPO_EPOCHS:           int   = 13
MAX_GRAD_NORM:        float = 0.5
ENTROPY_FACTOR:       float = 0.0
ACTOR_LEARNING_RATE:  float = 3.5e-4
CRITIC_LEARNING_RATE: float = 1.5e-3
RECURRENT_SEQ_LEN:    int = 1
RECURRENT_LAYERS:     int = 1
ROLLOUT_STEPS:        int = 2048
PARALLEL_ROLLOUTS:    int = 8
PATIENCE:             int = int(1e6)
TRAINABLE_STD_DEV:    bool = False
INIT_LOG_STD_DEV:     float = 0.0
NUMBER_OF_MIXTURES: float = 5
AIS_SIZE: int = HIDDEN_SIZE//2
KERNEL = "gaussian"

parser = argparse.ArgumentParser()

parser.add_argument("--ENV", default='Hopper-v2', type=str, help='Environment_id')
parser.add_argument("-n", "--name", default=1, type=int, help="Name of the run")
parser.add_argument("--nlin", default='tanh', type=str, help="Type of activation function")
parser.add_argument("--ckpt_freq", default=1000, type=int, help="Checkpoint frequency")
parser.add_argument("--dir", default='BaselineTest', type=str, help="folder name for saving the results")
parser.add_argument("--exp_name", default='Hopper', type=str, help="directory for experiment name")
parser.add_argument("--kernel", default=KERNEL, type=str, help="MMD Kernel")
# parser.add_argument("-n", "--name", required=True, type=int, help="Name of the run")
# parser.add_argument("--dir", required=True, type=str, help="folder name for saving the results")
# parser.add_argument("--exp_name", required=True, type=str, help="directory for experiment name")
parser.add_argument('--max_iterations', default=int(1.5e3), type=int, help='Total run iterations')
parser.add_argument('--rollout_steps', default=ROLLOUT_STEPS, type=int, help='Number of steps takes by the policy')
parser.add_argument("--gae_lambda", default=GAE_LAMBDA, type=float, help="Lambda")
parser.add_argument("--ppo_clip", default=PPO_CLIP, type=float, help="cliping range")
parser.add_argument("--no_of_mixtures", default=NUMBER_OF_MIXTURES, type=int, help='Number of Gaussians')
parser.add_argument("--ais_size", default=AIS_SIZE, type=int, help='Dimension of AIS.')
parser.add_argument("--entropy_factor", default=ENTROPY_FACTOR, type=float, help='Entropy Coefficient')
parser.add_argument('--seed', default=0, type=int, help='Initial seed')
parser.add_argument('--actor_learning_rate', default=ACTOR_LEARNING_RATE, type=float, help='Actor learning rate')
parser.add_argument('--critic_learning_rate', default=CRITIC_LEARNING_RATE, type=float, help='Critic learning rate')
parser.add_argument('--psi_learning_rate', default=1.5*ACTOR_LEARNING_RATE, type=float, help='Ais learning rate')
parser.add_argument('--hidden_size', default=HIDDEN_SIZE, type=int, help='Hidden state dimension')
parser.add_argument('--parallel_rollouts', default=PARALLEL_ROLLOUTS, type=int, help='Number of parallel rollouts')
parser.add_argument('--patience', default=int(1e7), type=int, help='Stopping condition')
parser.add_argument('--recurrent_seq_len', default=RECURRENT_SEQ_LEN, type=int, help='Recurrent sequence used for training')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Batch size for training')
parser.add_argument('--recurrent_layers', default=RECURRENT_LAYERS, type=int, help='Recurrent Layers in RNN')
parser.add_argument('--scale_reward', default=SCALE_REWARD, type=float, help='Reward Scale')
parser.add_argument('--discount', default=DISCOUNT, type=float, help='Discount factor')
parser.add_argument('--ppo_epochs', default=PPO_EPOCHS, type=float, help='Epochs for PPO')
parser.add_argument('--max_grad_norm', default=MAX_GRAD_NORM, type=float, help='Max norm for the gradient')
parser.add_argument('--trainable_std_dev', default='store_false', type=bool, help='Std deviation for policy')
parser.add_argument('--init_log_std_dev', default=INIT_LOG_STD_DEV, type=float, help='Initial std deviation for policy')
args = parser.parse_args()


@dataclass
class HyperParameters():
    ENV: str = args.ENV
    dir: str = args.dir
    exp_name: str = args.exp_name
    name: str = args.name
    nlin: str = args.nlin
    scale_reward: float = args.scale_reward
    hidden_size: int = args.hidden_size
    batch_size: int = args.batch_size
    discount: float = args.discount
    gae_lambda: float = args.gae_lambda
    ppo_clip: float = args.ppo_clip
    ppo_epochs: int = args.ppo_epochs
    max_grad_norm: float = args.max_grad_norm
    entropy_factor: float = args.entropy_factor
    actor_learning_rate: float = args.actor_learning_rate
    critic_learning_rate: float = args.critic_learning_rate
    recurrent_seq_len: int = args.recurrent_seq_len
    recurrent_layers: int = args.recurrent_layers
    rollout_steps: int = args.rollout_steps
    parallel_rollouts: int = args.parallel_rollouts
    patience: int = args.patience
    number_of_mixtures: int = args.no_of_mixtures
    ais_size: int = args.ais_size
    max_iterations: int = args.max_iterations
    # Apply to continuous action spaces only
    trainable_std_dev: bool = args.trainable_std_dev
    init_log_std_dev: float = args.init_log_std_dev
    min_reward: float = MIN_REWARD
    kernel: str = args.kernel
    ckpt_freq: int = args.ckpt_freq
    psi_learning_rate: float = args.psi_learning_rate
