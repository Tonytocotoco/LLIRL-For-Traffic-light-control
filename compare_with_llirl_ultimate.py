"""
Script so sánh DDQN, LLIRL Ultimate (và PPO nếu có)
- LLIRL: chọn policy bằng general policy + E-step (env_models + CRP)
- So sánh thêm total waiting time, average vehicles per episode bằng line chart
"""

import sys
import os
import gym
import numpy as np
import argparse
import torch
import torch.serialization
from tqdm import tqdm
import time
import json
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


torch.serialization.add_safe_globals([np.ndarray, np.core.multiarray._reconstruct])
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# Add paths
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, 'ddqn_sumo'))
sys.path.insert(0, os.path.join(BASE_DIR, 'llirl_sumo'))
sys.path.insert(0, os.path.join(BASE_DIR, 'ppo_sumo'))

# Register environments
from ddqn_sumo.myrllib.envs import sumo_env as ddqn_env  # noqa: F401
from llirl_sumo.myrllib.envs import sumo_env as llirl_env_eval  # noqa: F401
from ppo_sumo.myrllib.envs import sumo_env as ppo_env  # noqa: F401
import gym

# Import traci for connection management
try:
    import traci
    import traci.exceptions
except ImportError:
    traci = None

gym.register(
    'SUMO-SingleIntersection-DDQN-v1',
    entry_point='ddqn_sumo.myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600

)

gym.register(
    'SUMO-SingleIntersection-LLIRL-v1',
    entry_point='llirl_sumo.myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600
)

gym.register(
    'SUMO-SingleIntersection-PPO-v1',
    entry_point='ppo_sumo.myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=3600
)

# Import models & utils
from ddqn_sumo.myrllib.algorithms.ddqn import DDQN
from llirl_sumo.load_models import load_crp_state, load_task_info   
from llirl_sumo.myrllib.utils.policy_utils import create_general_policy
from llirl_sumo.myrllib.samplers.sampler import BatchSampler
from ppo_sumo.myrllib.policies import NormalMLPPolicy as PPO_NormalMLPPolicy
from llirl_sumo.myrllib.policies import NormalMLPPolicy as LLIRL_NormalMLPPolicy  


# env models + likelihood tools (E-step)
from llirl_sumo.myrllib.mixture.env_model import EnvModel, construct_env_io
from llirl_sumo.myrllib.mixture.inference import compute_likelihood

start_time = time.time()

######################## Arguments ############################################

parser = argparse.ArgumentParser(
    description='Compare DDQN, LLIRL Ultimate và PPO (nếu có)'
)
parser.add_argument(
    '--sumo_config', type=str,
    default='nets/120p4k/run_120p4k.sumocfg',
    help='Path đến file SUMO .sumocfg'
)
parser.add_argument(
    '--ddqn_model_path', type=str,
    default='ddqn_sumo/saves/sumo_single_intersection/ddqn_model_final.pth',
    help='Path đến model DDQN'
)

parser.add_argument(
    '--llirl_ultimate_path', type=str,
    default='llirl_sumo/saves/120p4k_ultimate_test2',
    help='Path đến thư mục LLIRL Ultimate (saves/<run_name>)'
)
parser.add_argument(
    '--ppo_model_path', type=str,
    default='ppo_sumo/output/models/ppo_final.pth',
    help='Path đến model PPO (nếu có, để so sánh thêm)'
)

parser.add_argument(
    '--output', type=str,
    default='output/comparison_with_llirl_ultimate',
    help='Thư mục output để lưu kết quả và hình'
)
parser.add_argument(
    '--num_episodes', type=int,
    default=4,
    help='Số episode test cho mỗi model'
)
parser.add_argument(
    '--max_steps', type=int,
    default=7200,
    help='Số step tối đa mỗi episode'
)
parser.add_argument(
    '--device', type=str,
    default='cpu',
    help='cpu hoặc cuda'
)
parser.add_argument(
    '--seed', type=int,
    default=20,
    help='Random seed'
)

args = parser.parse_args()

print("=" * 80)
print("MODEL COMPARISON: DDQN vs LLIRL ULTIMATE (và PPO nếu có)")
print("=" * 80)
print(f"SUMO Config      : {args.sumo_config}")
print(f"DDQN Model       : {args.ddqn_model_path}")
print(f"LLIRL Ultimate   : {args.llirl_ultimate_path}")
print(f"PPO Model (opt.) : {args.ppo_model_path}")
print(f"Num episodes     : {args.num_episodes}")
print(f"Device           : {args.device}")
print("=" * 80)

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output, exist_ok=True)

######################## Load Models ##########################################

print("\n" + "=" * 80)
print("LOADING MODELS")
print("=" * 80)

sumo_config_path = os.path.abspath(args.sumo_config)

# -------- DDQN --------
print("\n[1/3] Loading DDQN model...")
ddqn_env_test = gym.make(
    'SUMO-SingleIntersection-DDQN-v1',
    sumo_config_path=sumo_config_path,
    max_steps=args.max_steps
)

state_dim = ddqn_env_test.observation_space.shape[0]
action_dim = ddqn_env_test.action_space.n
print(f"State dim: {state_dim}, Action dim: {action_dim}")

ddqn = DDQN(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device
)

if not os.path.exists(args.ddqn_model_path):
    raise FileNotFoundError(f"DDQN model not found at {args.ddqn_model_path}")

ddqn.load(args.ddqn_model_path)
ddqn.q_network.eval()
ddqn.target_network.eval()
print(f"✓ DDQN model loaded from {args.ddqn_model_path}")
ddqn_env_test.close()

# -------- LLIRL Ultimate --------
print("\n[2/3] Loading LLIRL Ultimate model...")
if not os.path.exists(args.llirl_ultimate_path):
    raise FileNotFoundError(
        f"LLIRL Ultimate model directory not found at {args.llirl_ultimate_path}"
    )
def load_policies(model_dir, device):
    """
    Load LLIRL policies từ policies_final.pth
    - Trả về: (list[policy], checkpoint_dict)
    """
    policies_path = os.path.join(model_dir, 'policies_final.pth')
    if not os.path.exists(policies_path):
        raise FileNotFoundError(f"policies_final.pth not found in {model_dir}")

    print(f"[LLIRL] Loading policies from {policies_path} (weights_only=False)...")
    ckpt = torch.load(
        policies_path,
        map_location=device,
        weights_only=False
    )

    # Lấy thông tin kiến trúc từ checkpoint (giống training)
    state_dim = ckpt.get('state_dim')
    action_dim = ckpt.get('action_dim')
    hidden_size = ckpt.get('hidden_size', 200)
    num_layers = ckpt.get('num_layers', 2)

    if state_dim is None or action_dim is None:
        raise ValueError("Checkpoint missing state_dim/action_dim in policies_final.pth")

    policy_state_dicts = ckpt.get('policies', [])
    llirl_policies = []

    for i, sd in enumerate(policy_state_dicts):
        if sd is None:
            llirl_policies.append(None)
            continue

        # Tạo policy LLIRL đúng kiến trúc rồi load state_dict
        policy = LLIRL_NormalMLPPolicy(
            state_dim,
            action_dim,
            hidden_sizes=(hidden_size,) * num_layers
        ).to(device)

        policy.load_state_dict(sd)
        policy.eval()
        llirl_policies.append(policy)

    print(f"[LLIRL] Loaded {len(llirl_policies)} policies (num_policies={ckpt.get('num_policies')})")
    return llirl_policies, ckpt

llirl_policies, llirl_checkpoint = load_policies(args.llirl_ultimate_path, device=device)
print(f"✓ Loaded {len(llirl_policies)} LLIRL Ultimate policies")

# Task & CRP
crp = load_crp_state(args.llirl_ultimate_path)


# -------- Env models for E-step --------
print("\n[2b] Loading environment models for LLIRL E-step selection...")
env_models_path = os.path.join(args.llirl_ultimate_path, 'env_models.pth')
env_models = []
clustering_config = {
    'env_type': 'reward',
    'H': 4,
    'sigma': 0.1
}
if os.path.exists(env_models_path):
    env_models_ckpt = torch.load(env_models_path, map_location=device)
    input_size = env_models_ckpt['input_size']
    output_size = env_models_ckpt['output_size']
    hidden_sizes = env_models_ckpt.get(
        'hidden_sizes',
        (env_models_ckpt.get('hidden_size', 200),)
    )
    for state_dict in env_models_ckpt['env_models']:
        m = EnvModel(input_size, output_size, hidden_sizes=hidden_sizes).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        env_models.append(m)
    print(f"✓ Loaded {len(env_models)} environment models for clustering")
else:
    print(f"[WARNING] env_models.pth not found at {env_models_path}, "
          f"E-step selection will be disabled")

# Load clustering hyperparameters (env_type, H, sigma)
clustering_summary_path = os.path.join(args.llirl_ultimate_path, 'clustering_summary.json')
if os.path.exists(clustering_summary_path):
    with open(clustering_summary_path, 'r') as f:
        cs = json.load(f)
    clustering_config['env_type'] = cs.get('env_type', clustering_config['env_type'])
    clustering_config['H'] = cs.get('H', clustering_config['H'])
    clustering_config['sigma'] = cs.get('sigma', clustering_config['sigma'])
    print(f"✓ Loaded clustering config: env_type={clustering_config['env_type']}, "
          f"H={clustering_config['H']}, sigma={clustering_config['sigma']}")
else:
    print("[INFO] clustering_summary.json not found, using default clustering config")

# -------- Helper: auto output path --------
def auto_detect_output_path(model_path, folder_name):
    """Auto-detect output path from model path"""
    if 'saves' in model_path:
        return model_path.replace('saves', 'output')
    elif 'output' in model_path:
        return model_path  # already an output path
    else:
        model_dir_name = os.path.basename(model_path)
        if folder_name == 'ddqn_sumo':
            return os.path.join('ddqn_sumo', 'output', model_dir_name)
        if folder_name == 'ppo_sumo':
            return os.path.join('ppo_sumo', 'output', model_dir_name)
        if folder_name == 'llirl_sumo':
            return os.path.join('llirl_sumo', 'output', model_dir_name)
    return None

ddqn_output_path = auto_detect_output_path(args.ddqn_model_path, 'ddqn_sumo') \
                   or 'ddqn_sumo/output/120p4k'
ppo_output_path = auto_detect_output_path(args.ppo_model_path, 'ppo_sumo') \
                  or 'ppo_sumo/output/120p4k'
llirl_output_path = auto_detect_output_path(args.llirl_ultimate_path, 'llirl_sumo') \
                    or 'llirl_sumo/output/120p4k_ultimate'


# -------- PPO model (optional) --------
print("\n[3/3] Loading PPO model (optional)...")
ppo_env_test = gym.make(
    'SUMO-SingleIntersection-PPO-v1',
    sumo_config_path=sumo_config_path,
    max_steps=args.max_steps
)

ppo_state_dim = int(np.prod(ppo_env_test.observation_space.shape))
ppo_action_dim = int(np.prod(ppo_env_test.action_space.shape))
print(f"PPO State dim: {ppo_state_dim}, Action dim: {ppo_action_dim}")

ppo_model_path = args.ppo_model_path
ppo_policy = None
ppo_metrics = None

if os.path.exists(ppo_model_path):
    print(os.path.exists(ppo_model_path))
    try:
        checkpoint = torch.load(ppo_model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
            print(f"[WARNING] {ppo_model_path} chứa DDQN, bỏ qua PPO.")
            ppo_model_path = None
        else:
            hidden_size = 200
            num_layers = 2
            ppo_policy = PPO_NormalMLPPolicy(
                ppo_state_dim,
                ppo_action_dim,
                hidden_sizes=(hidden_size,) * num_layers
            ).to(device)

            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                state_dict = checkpoint['policy']
            else:
                state_dict = checkpoint

            ppo_policy.load_state_dict(state_dict)
            ppo_policy.eval()
            print(f"✓ PPO model loaded from {ppo_model_path}")
    except Exception as e:
        print(f"[WARNING] Failed to load PPO model: {e}")
        ppo_model_path = None
        ppo_policy = None
else:
    print("[INFO] PPO model path not found, skip PPO comparison.")

ppo_env_test.close()

######################## LLIRL E-step selection ###############################

def build_general_llirl_policy(llirl_policies, crp, checkpoint, device):
    """Tạo general policy từ library LLIRL + CRP prior (dùng để thu thập 1 episode)."""
    valid_policies = [p for p in llirl_policies if p is not None]
    if len(valid_policies) == 0:
        raise ValueError("No valid LLIRL policies available to build general policy")

    state_dim = checkpoint.get('state_dim')
    action_dim = checkpoint.get('action_dim')
    hidden_size = checkpoint.get('hidden_size', 200)
    num_layers = checkpoint.get('num_layers', 2)

    if state_dim is None or action_dim is None:
        raise ValueError("Checkpoint missing state_dim/action_dim for general policy")

    if crp is not None and hasattr(crp, '_prior') and len(crp._prior) >= len(valid_policies):
        prior_slice = crp._prior[:len(valid_policies)]
        priors = prior_slice.tolist() if isinstance(prior_slice, np.ndarray) else list(prior_slice)
    else:
        priors = [1.0 / len(valid_policies)] * len(valid_policies)

    general_policy = create_general_policy(
        valid_policies,
        priors,
        state_dim,
        action_dim,
        (hidden_size,) * num_layers,
        device=device
    )
    general_policy.eval()
    return general_policy


def infer_cluster_with_E_step(llirl_sampler, general_policy, env_models, crp, clustering_config, device):

    if len(env_models) == 0:
        raise ValueError("No environment models loaded for E-step cluster inference")

    print("\n[LLIRL] Collecting 1 episode with general policy for E-step...")
    episodes = llirl_sampler.sample(general_policy, device=device)
    episodes_list = [episodes]

    inputs, outputs = construct_env_io(
        episodes_list,
        env_type=clustering_config.get('env_type', 'reward'),
        H=clustering_config.get('H', 4)
    )

    sigma = clustering_config.get('sigma', 0.1)
    num_models = len(env_models)

    logll = np.zeros(num_models)
    for idx in range(num_models):
        logll[idx] = compute_likelihood(env_models[idx], inputs, outputs, sigma=sigma)

    if crp is not None and hasattr(crp, '_prior') and len(crp._prior) >= num_models:
        prior_slice = crp._prior[:num_models]
    else:
        prior_slice = np.ones(num_models) / num_models

    prior_safe = np.clip(prior_slice, 1e-12, 1.0)
    logprior = np.log(prior_safe)

    log_post = logll + logprior
    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    posterior /= posterior.sum()

    cluster_id = int(np.argmax(posterior)) + 1

    print("\n[LLIRL E-step inference]")
    print(f"  log-likelihoods: {logll}")
    print(f"  prior:          {prior_safe}")
    print(f"  posterior:      {posterior}")
    print(f"  → Selected cluster: {cluster_id}")

    return cluster_id

######################## Evaluation Functions ##################################

def evaluate_ddqn(env, model, num_episodes, max_steps):
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'vehicle_counts': [],
        'speeds': [],
        'total_waiting_times': []
    }

    print(f"\nEvaluating DDQN for {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes), desc="DDQN episodes"):
        if hasattr(env.unwrapped, 'reset_task'):
            env.unwrapped.reset_task(task[ep])
        state = env.reset()
        episode_reward = 0
        episode_waiting, episode_queues = [], []
        episode_vehicles, episode_speeds = [], []

        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model.q_network(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            if 'waiting_time' in info:
                episode_waiting.append(info['waiting_time'])
            if 'queue_length' in info:
                episode_queues.append(info['queue_length'])
            if 'vehicle_count' in info:
                episode_vehicles.append(info['vehicle_count'])
            if 'speed' in info:
                episode_speeds.append(info['speed'])

            state = next_state
            if done:
                break

        metrics['rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        if episode_waiting:
            ew = np.array(episode_waiting)
            metrics['waiting_times'].append(float(ew.mean()))
            metrics['total_waiting_times'].append(float(ew.sum()))
        if episode_queues:
            metrics['queue_lengths'].append(float(np.mean(episode_queues)))
        if episode_vehicles:
            metrics['vehicle_counts'].append(float(np.mean(episode_vehicles)))
        if episode_speeds:
            metrics['speeds'].append(float(np.mean(episode_speeds)))
    return metrics


def evaluate_llirl(env, policy, num_episodes, max_steps, device):
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'vehicle_counts': [],
        'speeds': [],
        'total_waiting_times': []
    }

    print(f"\nEvaluating LLIRL Ultimate for {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes), desc="LLIRL Ultimate episodes"):
        if hasattr(env.unwrapped, 'reset_task'):
            env.unwrapped.reset_task(task[ep])

        obs = env.reset()
        episode_reward = 0
        episode_waiting, episode_queues = [], []
        episode_vehicles, episode_speeds = [], []

        for step in range(max_steps):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                pi = policy(obs_tensor)
                action_tensor = pi.mean
                action = action_tensor.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]
                action = action.flatten()
                if len(action) != 4:
                    green_time_min = getattr(env.unwrapped, 'green_time_min', 10)
                    if len(action) == 1:
                        action = np.repeat(action, 4)
                    else:
                        action = (action[:4] if len(action) > 4
                                  else np.pad(action, (0, 4 - len(action)),
                                              'constant',
                                              constant_values=green_time_min))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            if 'waiting_time' in info:
                episode_waiting.append(info['waiting_time'])
            if 'queue_length' in info:
                episode_queues.append(info['queue_length'])
            if 'vehicle_count' in info:
                episode_vehicles.append(info['vehicle_count'])
            if 'speed' in info:
                episode_speeds.append(info['speed'])

            obs = next_obs
            if done:
                break

        metrics['rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        if episode_waiting:
            ew = np.array(episode_waiting)
            metrics['waiting_times'].append(float(ew.mean()))
            metrics['total_waiting_times'].append(float(ew.sum()))
        if episode_queues:
            metrics['queue_lengths'].append(float(np.mean(episode_queues)))
        if episode_vehicles:
            metrics['vehicle_counts'].append(float(np.mean(episode_vehicles)))
        if episode_speeds:
            metrics['speeds'].append(float(np.mean(episode_speeds)))
    return metrics


def evaluate_ppo(env, policy, num_episodes, max_steps, device):
    if policy is None:
        return None

    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'vehicle_counts': [],
        'speeds': [],
        'total_waiting_times': []
    }

    print(f"\nEvaluating PPO for {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes), desc="PPO episodes"):
        if hasattr(env.unwrapped, 'reset_task'):
            env.unwrapped.reset_task(task[ep])
            
        obs = env.reset()
        episode_reward = 0
        episode_waiting, episode_queues = [], []
        episode_vehicles, episode_speeds = [], []

        for step in range(max_steps):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                # pi = policy(obs_tensor)
                # action_tensor = pi.mean
                # action = action_tensor.cpu().numpy()
                dist = policy(obs_tensor)
                action_tensor = dist.sample()
                action = action_tensor.cpu().numpy()

                if action.ndim > 1:
                    action = action[0]
                action = action.flatten()
                if len(action) != 4:
                    green_time_min = getattr(env.unwrapped, 'green_time_min', 10)
                    if len(action) == 1:
                        action = np.repeat(action, 4)
                    else:
                        action = (action[:4] if len(action) > 4
                                  else np.pad(action, (0, 4 - len(action)),
                                              'constant',
                                              constant_values=green_time_min))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            if 'waiting_time' in info:
                episode_waiting.append(info['waiting_time'])
            if 'queue_length' in info:
                episode_queues.append(info['queue_length'])
            if 'vehicle_count' in info:
                episode_vehicles.append(info['vehicle_count'])
            if 'speed' in info:
                episode_speeds.append(info['speed'])

            obs = next_obs
            if done:
                break

        metrics['rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        if episode_waiting:
            ew = np.array(episode_waiting)
            metrics['waiting_times'].append(float(ew.mean()))
            metrics['total_waiting_times'].append(float(ew.sum()))
        if episode_queues:
            metrics['queue_lengths'].append(float(np.mean(episode_queues)))
        if episode_vehicles:
            metrics['vehicle_counts'].append(float(np.mean(episode_vehicles)))
        if episode_speeds:
            metrics['speeds'].append(float(np.mean(episode_speeds)))
    return metrics

######################## Run Comparison #######################################

print("\n" + "=" * 80)
print("RUNNING COMPARISON")
print("=" * 80)

# Task cho test
print("\nGenerating RANDOM test tasks (intensity, var) like env_clustering.py...")

task = np.hstack([
    np.random.uniform(1.5, 2.5, size=(args.num_episodes, 1)),
    np.random.uniform(0.0, 0.5, size=(args.num_episodes, 1))
]).tolist()


print("Random test tasks:")
for i, t in enumerate(task):
    print(f"  Episode {i+1}: intensity={t[0]:.2f}, var={t[1]:.2f}")


# Sampler LLIRL
llirl_sampler = BatchSampler(
    'SUMO-SingleIntersection-LLIRL-v1',
    batch_size=1,
    num_workers=0,
    seed=args.seed,
    sumo_config_path=sumo_config_path
)
if task is not None:
    first_task = [float(task[0][0]), float(task[0][1])]
    llirl_sampler.reset_task(first_task)
# Chọn policy LLIRL bằng general policy + E-step (nếu có env_models & CRP)
# Chọn policy LLIRL bằng general policy + E-step (nếu có env_models & CRP)
used_cluster_id = None
valid_llirl_policies = [p for p in llirl_policies if p is not None]

if len(env_models) > 0 and crp is not None:
    print("\n[LLIRL] Selecting policy with general policy + E-step cluster inference...")
    try:
        # Build general policy
        general_policy = build_general_llirl_policy(llirl_policies, crp, llirl_checkpoint, device)

        # E-step
        inferred_cluster_id = infer_cluster_with_E_step(
            llirl_sampler, general_policy, env_models, crp, clustering_config, device
        )
        cluster_index = inferred_cluster_id - 1

        # Get policy follow cluster
        if (cluster_index < 0 or
                cluster_index >= len(llirl_policies) or
                llirl_policies[cluster_index] is None):
            print(f"[WARNING] Cluster {inferred_cluster_id} không có policy hợp lệ, "
                  f"sẽ chọn RANDOM policy trong thư viện.")
            if not valid_llirl_policies:
                raise ValueError("No LLIRL policies loaded!")
            llirl_policy = random.choice(valid_llirl_policies)
            used_cluster_id = -1   # -1 = random
        else:
            llirl_policy = llirl_policies[cluster_index]
            used_cluster_id = inferred_cluster_id

    except Exception as e:
        # If General_policy none
        print(f"[WARNING] Không thể sử dụng general policy / E-step: {e}")
        print("[LLIRL] → Chọn RANDOM policy trong thư viện LLIRL để evaluate.")
        if not valid_llirl_policies:
            raise ValueError("No LLIRL policies loaded!")
        llirl_policy = random.choice(valid_llirl_policies)
        used_cluster_id = -1

    print(f"[LLIRL] Using policy (cluster_id={used_cluster_id}) for evaluation")

else:
    # Không có env_models hoặc CRP → không làm E-step, chỉ chọn random
    print("\n[LLIRL] Env models hoặc CRP không có, chọn RANDOM policy trong thư viện.")
    if not valid_llirl_policies:
        raise ValueError("No LLIRL policies loaded!")
    llirl_policy = random.choice(valid_llirl_policies)
    used_cluster_id = -1

llirl_policy.eval()
llirl_sampler._env.close()

# --- Evaluate DDQN ---
print("\n" + "=" * 80)
print("[1/3] Evaluating DDQN model...")
print("=" * 80)

ddqn_env = gym.make(
    'SUMO-SingleIntersection-DDQN-v1',
    sumo_config_path=sumo_config_path,
    max_steps=args.max_steps
)
print("DDQN internal max_steps:", getattr(ddqn_env.unwrapped, "max_steps", None))
ddqn_metrics = evaluate_ddqn(ddqn_env, ddqn, args.num_episodes, args.max_steps)
ddqn_env.close()

if traci is not None:
    try:
        traci.close()
    except Exception:
        pass
time.sleep(1.0)

# --- Evaluate LLIRL ---
print("\n" + "=" * 80)
print("[2/3] Evaluating LLIRL Ultimate model...")
print("=" * 80)

llirl_env_eval = gym.make(
    'SUMO-SingleIntersection-LLIRL-v1',
    sumo_config_path=sumo_config_path,
    max_steps=args.max_steps
)


llirl_metrics = evaluate_llirl(llirl_env_eval, llirl_policy, args.num_episodes, args.max_steps, device)
llirl_env_eval.close()

if traci is not None:
    try:
        traci.close()
    except Exception:
        pass
time.sleep(1.0)

# --- Evaluate PPO (optional) ---

if ppo_policy is not None:
    print("\n" + "=" * 80)
    print("[3/3] Evaluating PPO model...")
    print("=" * 80)

    ppo_env = gym.make(
        'SUMO-SingleIntersection-PPO-v1',
        sumo_config_path=sumo_config_path,
        max_steps=args.max_steps
    )


    ppo_metrics = evaluate_ppo(ppo_env, ppo_policy, args.num_episodes, args.max_steps, device)
    ppo_env.close()

    if traci is not None:
        try:
            traci.close()
        except Exception:
            pass
else:
    print("\n[3/3] Skipping PPO evaluation (no PPO model)")

if traci is not None:
    try:
        traci.close()
    except Exception:
        pass

######################## Results Analysis ######################################

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

def print_metrics(name, metrics):
    if metrics is None:
        print(f"\n{name}: Not evaluated (model not found)")
        return
    print(f"\n{name}:")
    print(f"  Average Reward: {np.mean(metrics['rewards']):.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"  Average Episode Length: {np.mean(metrics['episode_lengths']):.1f} ± "
          f"{np.std(metrics['episode_lengths']):.1f}")
    if metrics['waiting_times']:
        print(f"  Average Waiting Time: {np.mean(metrics['waiting_times']):.2f} ± "
              f"{np.std(metrics['waiting_times']):.2f} seconds")
    if metrics['total_waiting_times']:
        print(f"  Total Waiting Time / Episode: {np.mean(metrics['total_waiting_times']):.2f} ± "
              f"{np.std(metrics['total_waiting_times']):.2f}")
    if metrics['queue_lengths']:
        print(f"  Average Queue Length: {np.mean(metrics['queue_lengths']):.2f} ± "
              f"{np.std(metrics['queue_lengths']):.2f} vehicles")
    if metrics['vehicle_counts']:
        print(f"  Average Vehicle Count: {np.mean(metrics['vehicle_counts']):.2f} ± "
              f"{np.std(metrics['vehicle_counts']):.2f}")
    if metrics['speeds']:
        print(f"  Average Speed: {np.mean(metrics['speeds']):.2f} ± "
              f"{np.std(metrics['speeds']):.2f} m/s")

print_metrics("DDQN", ddqn_metrics)
print_metrics("LLIRL Ultimate", llirl_metrics)
print_metrics("PPO", ppo_metrics)

print("\n" + "-" * 80)
print("COMPARISON SUMMARY:")
print("-" * 80)

models_metrics = {
    'DDQN': ddqn_metrics,
    'LLIRL Ultimate': llirl_metrics,
    'PPO': ppo_metrics
}
valid_models = {k: v for k, v in models_metrics.items() if v is not None}

avg_rewards = {}
best_model = None
if valid_models:
    avg_rewards = {k: np.mean(v['rewards']) for k, v in valid_models.items()}
    best_model = max(avg_rewards, key=avg_rewards.get)
    worst_model = min(avg_rewards, key=avg_rewards.get)

    print(f"\n🏆 Best Model (by Average Reward): {best_model}")
    print(f"  Average Reward: {avg_rewards[best_model]:.2f}")

    print(f"\n📊 Ranking:")
    for model_name, reward in sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True):
        improvement = ((reward - avg_rewards[worst_model]) / abs(avg_rewards[worst_model])) * 100 \
            if avg_rewards[worst_model] != 0 else 0
        vs_ddqn = ((reward - avg_rewards.get('DDQN', reward)) /
                   abs(avg_rewards.get('DDQN', reward))) * 100 if avg_rewards.get('DDQN') else 0
        print(f"  {model_name}: {reward:.2f}")
        if model_name != 'DDQN' and 'DDQN' in avg_rewards:
            print(f"    → vs DDQN: {vs_ddqn:+.1f}%")

if valid_models and all(m['waiting_times'] for m in valid_models.values()):
    print(f"\n⏱️  Waiting Time Comparison (lower is better):")
    avg_waiting = {k: np.mean(v['waiting_times']) for k, v in valid_models.items()}
    worst_waiting = max(avg_waiting, key=avg_waiting.get)
    for model_name, waiting in sorted(avg_waiting.items(), key=lambda x: x[1]):
        improvement = ((avg_waiting[worst_waiting] - waiting) / avg_waiting[worst_waiting]) * 100 \
            if avg_waiting[worst_waiting] != 0 else 0
        print(f"  {model_name}: {waiting:.2f}s ({improvement:+.1f}% vs worst)")

if valid_models and all(m['queue_lengths'] for m in valid_models.values()):
    print(f"\n Queue Length Comparison (lower is better):")
    avg_queues = {k: np.mean(v['queue_lengths']) for k, v in valid_models.items()}
    worst_queues = max(avg_queues, key=avg_queues.get)
    for model_name, q in sorted(avg_queues.items(), key=lambda x: x[1]):
        improvement = ((avg_queues[worst_queues] - q) / avg_queues[worst_queues]) * 100 \
            if avg_queues[worst_queues] != 0 else 0
        print(f"  {model_name}: {q:.2f} vehicles ({improvement:+.1f}% vs worst)")




######################## Save Results ##########################################

results = {
    'config': {
        'sumo_config': args.sumo_config,
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'seed': args.seed,
        'device': str(device),
        'llirl_ultimate_path': args.llirl_ultimate_path,
        'llirl_selected_cluster': int(used_cluster_id) if used_cluster_id is not None else None
    },
    'ddqn': {
        'model_path': args.ddqn_model_path,
        'output_path': ddqn_output_path,
        'rewards': ddqn_metrics['rewards'],
        'episode_lengths': ddqn_metrics['episode_lengths'],
        'waiting_times': ddqn_metrics['waiting_times'],
        'total_waiting_times': ddqn_metrics['total_waiting_times'],
        'queue_lengths': ddqn_metrics['queue_lengths'],
        'vehicle_counts': ddqn_metrics['vehicle_counts'],
        'speeds': ddqn_metrics['speeds'],
        'avg_reward': float(np.mean(ddqn_metrics['rewards'])),
        'avg_waiting_time': float(np.mean(ddqn_metrics['waiting_times'])) if ddqn_metrics['waiting_times'] else None,
        'avg_total_waiting_time': float(np.mean(ddqn_metrics['total_waiting_times'])) if ddqn_metrics['total_waiting_times'] else None,
        'avg_queue_length': float(np.mean(ddqn_metrics['queue_lengths'])) if ddqn_metrics['queue_lengths'] else None,
    },
    'llirl_ultimate': {
        'model_path': args.llirl_ultimate_path,
        'output_path': llirl_output_path,
        'rewards': llirl_metrics['rewards'],
        'episode_lengths': llirl_metrics['episode_lengths'],
        'waiting_times': llirl_metrics['waiting_times'],
        'total_waiting_times': llirl_metrics['total_waiting_times'],
        'queue_lengths': llirl_metrics['queue_lengths'],
        'vehicle_counts': llirl_metrics['vehicle_counts'],
        'speeds': llirl_metrics['speeds'],
        'avg_reward': float(np.mean(llirl_metrics['rewards'])),
        'avg_waiting_time': float(np.mean(llirl_metrics['waiting_times'])) if llirl_metrics['waiting_times'] else None,
        'avg_total_waiting_time': float(np.mean(llirl_metrics['total_waiting_times'])) if llirl_metrics['total_waiting_times'] else None,
        'avg_queue_length': float(np.mean(llirl_metrics['queue_lengths'])) if llirl_metrics['queue_lengths'] else None,

    },
    'ppo': {
        'model_path': ppo_model_path if ppo_policy is not None else None,
        'output_path': ppo_output_path if ppo_policy is not None else None,

        'rewards': ppo_metrics['rewards'] if ppo_metrics else None,
        'episode_lengths': ppo_metrics['episode_lengths'] if ppo_metrics else None,
        'waiting_times': ppo_metrics['waiting_times'] if ppo_metrics else None,
        'total_waiting_times': ppo_metrics['total_waiting_times'] if ppo_metrics else None,
        'queue_lengths': ppo_metrics['queue_lengths'] if ppo_metrics else None,
        'vehicle_counts': ppo_metrics['vehicle_counts'] if ppo_metrics else None,
        'speeds': ppo_metrics['speeds'] if ppo_metrics else None,
        'avg_reward': float(np.mean(ppo_metrics['rewards'])) if ppo_metrics else None,
        'avg_waiting_time': float(np.mean(ppo_metrics['waiting_times'])) if ppo_metrics and ppo_metrics['waiting_times'] else None,
        'avg_total_waiting_time': float(np.mean(ppo_metrics['total_waiting_times'])) if ppo_metrics and ppo_metrics['total_waiting_times'] else None,
        'avg_queue_length': float(np.mean(ppo_metrics['queue_lengths'])) if ppo_metrics and ppo_metrics['queue_lengths'] else None,

    },
    'comparison': {
        'best_model_by_reward': best_model if valid_models else None,
        'avg_rewards': {k: float(v) for k, v in avg_rewards.items()} if valid_models else None,
    }
}

results_path = os.path.join(args.output, 'comparison_with_llirl_ultimate.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {results_path}")

# Save eval reward arrays
np.save(os.path.join(args.output, 'ddqn_rewards.npy'), np.array(ddqn_metrics['rewards']))
np.save(os.path.join(args.output, 'llirl_ultimate_rewards.npy'), np.array(llirl_metrics['rewards']))
if ppo_metrics:
    np.save(os.path.join(args.output, 'ppo_rewards.npy'), np.array(ppo_metrics['rewards']))



######################## Plot Comparison Charts ################################

print("\n" + "=" * 80)
print("GENERATING COMPARISON CHARTS")
print("=" * 80)

def plot_comparison_charts(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10

    fig = plt.figure(figsize=(20, 14))
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # 1. Evaluation Rewards Comparison (Bar)
    ax1 = plt.subplot(3, 3, 1)
    models, rewards, errors = [], [], []
    if results['ddqn']['avg_reward'] is not None:
        models.append('DDQN')
        rewards.append(results['ddqn']['avg_reward'])
        errors.append(np.std(results['ddqn']['rewards']))
    if results['llirl_ultimate']['avg_reward'] is not None:
        models.append('LLIRL\nUltimate')
        rewards.append(results['llirl_ultimate']['avg_reward'])
        errors.append(np.std(results['llirl_ultimate']['rewards']))
    if results['ppo']['avg_reward'] is not None:
        models.append('PPO')
        rewards.append(results['ppo']['avg_reward'])
        errors.append(np.std(results['ppo']['rewards']))
    if models:
        bars = ax1.bar(models, rewards, yerr=errors, capsize=5,
                       color=colors[:len(models)], alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Evaluation Rewards Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, r in zip(bars, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{r:.0f}', ha='center', va='bottom', fontweight='bold')

    # 2. Reward Distribution (Box)
    ax2 = plt.subplot(3, 3, 2)
    data_to_plot, labels = [], []
    if results['ddqn']['rewards']:
        data_to_plot.append(results['ddqn']['rewards'])
        labels.append('DDQN')
    if results['llirl_ultimate']['rewards']:
        data_to_plot.append(results['llirl_ultimate']['rewards'])
        labels.append('LLIRL Ultimate')
    if results['ppo']['rewards']:
        data_to_plot.append(results['ppo']['rewards'])
        labels.append('PPO')
    if data_to_plot:
        ax2.boxplot(
            data_to_plot, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2)
        )
        ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax2.set_title('Reward Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Waiting Time (bar)
    ax3 = plt.subplot(3, 3, 3)
    models_wt, waiting_times, errors_wt = [], [], []
    if results['ddqn']['avg_waiting_time'] is not None:
        models_wt.append('DDQN')
        waiting_times.append(results['ddqn']['avg_waiting_time'])
        errors_wt.append(np.std(results['ddqn']['waiting_times']))
    if results['llirl_ultimate']['avg_waiting_time'] is not None:
        models_wt.append('LLIRL\nUltimate')
        waiting_times.append(results['llirl_ultimate']['avg_waiting_time'])
        errors_wt.append(np.std(results['llirl_ultimate']['waiting_times']))
    if results['ppo']['avg_waiting_time'] is not None:
        models_wt.append('PPO')
        waiting_times.append(results['ppo']['avg_waiting_time'])
        errors_wt.append(np.std(results['ppo']['waiting_times']))
    if models_wt:
        bars = ax3.bar(models_wt, waiting_times, yerr=errors_wt, capsize=5,
                       color=colors[:len(models_wt)], alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Average Waiting Time (s)', fontsize=12, fontweight='bold')
        ax3.set_title('Waiting Time Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, wt in zip(bars, waiting_times):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{wt:.2f}s', ha='center', va='bottom', fontweight='bold')

    # 4. Queue Length
    ax4 = plt.subplot(3, 3, 4)
    models_ql, queue_lengths, errors_ql = [], [], []
    if results['ddqn']['avg_queue_length'] is not None:
        models_ql.append('DDQN')
        queue_lengths.append(results['ddqn']['avg_queue_length'])
        errors_ql.append(np.std(results['ddqn']['queue_lengths']))
    if results['llirl_ultimate']['avg_queue_length'] is not None:
        models_ql.append('LLIRL\nUltimate')
        queue_lengths.append(results['llirl_ultimate']['avg_queue_length'])
        errors_ql.append(np.std(results['llirl_ultimate']['queue_lengths']))
    if results['ppo']['avg_queue_length'] is not None:
        models_ql.append('PPO')
        queue_lengths.append(results['ppo']['avg_queue_length'])
        errors_ql.append(np.std(results['ppo']['queue_lengths']))
    if models_ql:
        bars = ax4.bar(models_ql, queue_lengths, yerr=errors_ql, capsize=5,
                       color=colors[:len(models_ql)], alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Average Queue Length (vehicles)', fontsize=12, fontweight='bold')
        ax4.set_title('Queue Length Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, ql in zip(bars, queue_lengths):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{ql:.2f}', ha='center', va='bottom', fontweight='bold')


    # 6. Episode length
    ax6 = plt.subplot(3, 3, 6)
    models_el, episode_lengths, errors_el = [], [], []
    if results['ddqn']['episode_lengths']:
        models_el.append('DDQN')
        episode_lengths.append(np.mean(results['ddqn']['episode_lengths']))
        errors_el.append(np.std(results['ddqn']['episode_lengths']))
    if results['llirl_ultimate']['episode_lengths']:
        models_el.append('LLIRL\nUltimate')
        episode_lengths.append(np.mean(results['llirl_ultimate']['episode_lengths']))
        errors_el.append(np.std(results['llirl_ultimate']['episode_lengths']))
    if results['ppo']['episode_lengths']:
        models_el.append('PPO')
        episode_lengths.append(np.mean(results['ppo']['episode_lengths']))
        errors_el.append(np.std(results['ppo']['episode_lengths']))
    if models_el:
        bars = ax6.bar(models_el, episode_lengths, yerr=errors_el, capsize=5,
                       color=colors[:len(models_el)], alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax6.set_ylabel('Average Episode Length', fontsize=12, fontweight='bold')
        ax6.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        for bar, el in zip(bars, episode_lengths):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{el:.0f}', ha='center', va='bottom', fontweight='bold')

    # 7. Speed
    ax7 = plt.subplot(3, 3, 7)
    models_sp, speeds, errors_sp = [], [], []
    if results['ddqn']['speeds']:
        models_sp.append('DDQN')
        speeds.append(np.mean(results['ddqn']['speeds']))
        errors_sp.append(np.std(results['ddqn']['speeds']))
    if results['llirl_ultimate']['speeds']:
        models_sp.append('LLIRL\nUltimate')
        speeds.append(np.mean(results['llirl_ultimate']['speeds']))
        errors_sp.append(np.std(results['llirl_ultimate']['speeds']))
    if results['ppo']['speeds']:
        models_sp.append('PPO')
        speeds.append(np.mean(results['ppo']['speeds']))
        errors_sp.append(np.std(results['ppo']['speeds']))
    if models_sp:
        bars = ax7.bar(models_sp, speeds, yerr=errors_sp, capsize=5,
                       color=colors[:len(models_sp)], alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax7.set_ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
        ax7.set_title('Speed Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        for bar, sp in zip(bars, speeds):
            ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{sp:.2f}', ha='center', va='bottom', fontweight='bold')

    # 8. Vehicle count
    ax8 = plt.subplot(3, 3, 8)
    models_vc, vehicle_counts, errors_vc = [], [], []
    if results['ddqn']['vehicle_counts']:
        models_vc.append('DDQN')
        vehicle_counts.append(np.mean(results['ddqn']['vehicle_counts']))
        errors_vc.append(np.std(results['ddqn']['vehicle_counts']))
    if results['llirl_ultimate']['vehicle_counts']:
        models_vc.append('LLIRL\nUltimate')
        vehicle_counts.append(np.mean(results['llirl_ultimate']['vehicle_counts']))
        errors_vc.append(np.std(results['llirl_ultimate']['vehicle_counts']))
    if results['ppo']['vehicle_counts']:
        models_vc.append('PPO')
        vehicle_counts.append(np.mean(results['ppo']['vehicle_counts']))
        errors_vc.append(np.std(results['ppo']['vehicle_counts']))
    if models_vc:
        bars = ax8.bar(models_vc, vehicle_counts, yerr=errors_vc, capsize=5,
                       color=colors[:len(models_vc)], alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax8.set_ylabel('Average Vehicle Count', fontsize=12, fontweight='bold')
        ax8.set_title('Vehicle Count Comparison', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        for bar, vc in zip(bars, vehicle_counts):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{vc:.1f}', ha='center', va='bottom', fontweight='bold')

    # 9. Overall winner
    ax9 = plt.subplot(3, 3, 9)
    if results['comparison']['best_model_by_reward'] is not None:
        bm = results['comparison']['best_model_by_reward']
        ax9.text(0.5, 0.7, 'Best Model:', ha='center', va='center',
                 transform=ax9.transAxes, fontsize=16, fontweight='bold')
        ax9.text(0.5, 0.5, bm, ha='center', va='center',
                 transform=ax9.transAxes, fontsize=20, fontweight='bold',
                 color='#2ecc71' if 'LLIRL' in bm else '#3498db')
        ax9.text(0.5, 0.3,
                 f"Reward: {results['comparison']['avg_rewards'][bm]:.0f}",
                 ha='center', va='center', transform=ax9.transAxes, fontsize=14)
        ax9.axis('off')
        ax9.set_title('Overall Winner', fontsize=14, fontweight='bold')

    plt.suptitle('Model Comparison: DDQN vs LLIRL Ultimate vs PPO',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    chart_path = os.path.join(output_dir, 'comparison_charts.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison charts saved to {chart_path}")
    plt.close()
def plot_episode_line_charts(results, output_dir):
    """
    Biểu đồ dây chạc theo episode:
      - Total Waiting Time / Episode
      - Average Vehicle Count / Episode
    Tập trung so sánh DDQN vs LLIRL, nếu PPO có dữ liệu thì vẽ thêm.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if sns is not None:
        sns.set_style("whitegrid")

    # Chỉ lấy những model thực sự có dữ liệu
    model_defs = [
        ("DDQN", "ddqn"),
        ("LLIRL Ultimate", "llirl_ultimate"),
        ("PPO", "ppo"),
    ]
    models = []
    for name, key in model_defs:
        r = results.get(key, {})
        if r and r.get("total_waiting_times"):
            models.append((name, key))

    if not models:
        print("[INFO] No episode-wise metrics to plot for line charts.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1) Total waiting time per episode
    ax1 = axes[0]
    for name, key in models:
        series = results[key]["total_waiting_times"]
        y = np.asarray(series, dtype=float)
        x = np.arange(1, len(y) + 1)
        ax1.plot(x, y, marker='o', linewidth=2, markersize=4, label=name)

    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Waiting Time (sum over steps)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Waiting Time per Episode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # 2) Average vehicle count per episode
    ax2 = axes[1]
    for name, key in models:
        series = results[key]["vehicle_counts"]
        if not series:
            continue
        y = np.asarray(series, dtype=float)
        x = np.arange(1, len(y) + 1)
        ax2.plot(x, y, marker='o', linewidth=2, markersize=4, label=name)

    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Vehicle Count', fontsize=12, fontweight='bold')
    ax2.set_title('Average Vehicle Count per Episode', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    line_chart_path = os.path.join(output_dir, 'episode_line_metrics_ddqn_vs_llirl.png')
    plt.savefig(line_chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Episode-wise line charts saved to {line_chart_path}")
    plt.close()
# ======= Generate charts ========
plot_comparison_charts(results, args.output)
plot_episode_line_charts(results, args.output)


print("\n" + "=" * 80)
print(f"Comparison completed in {(time.time() - start_time) / 60:.2f} minutes")
print("=" * 80)
