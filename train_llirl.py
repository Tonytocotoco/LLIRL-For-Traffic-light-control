"""
Train LLIRL với TẤT CẢ optimizations (VERSION CLEAN – NO PERFORMANCE POLICY)
Ultimate LLIRL training with all advanced improvements
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    llirl_dir = os.path.join(base_dir, "llirl_sumo")
    
    # Environment configuration
    sumo_config = "../nets/120p4k/run_120p4k.sumocfg"
    model_path = "saves/120p4k_ultimate_test2"
    output_path = "output/120p4k_ultimate_test2"
    
    print("=" * 80)
    print("LLIRL ULTIMATE TRAINING (CLEAN VERSION – NO PERFORMANCE POLICY)")
    print("=" * 80)
    print(f"SUMO Config: {sumo_config}")
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")
    print("=" * 80)
    
    # Step 1: Environment Clustering
    print("\n" + "=" * 80)
    print("Step 1: Environment Clustering (ULTIMATE)")
    print("=" * 80)
    
    clustering_cmd = [
        sys.executable, "env_clustering.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--et_length", "1",
        "--num_periods", "10",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "16",
        "--env_num_layers", "3",
        "--env_hidden_size", "256",
        "--H", "4",
        "--max_steps", "7200",
        "--zeta", "1",
        "--sigma", "0.25",
        "--tau1", "0.5",
        "--tau2", "0.5",
        "--em_steps", "10"
    ]
    
    subprocess.run(clustering_cmd, cwd=llirl_dir, check=True)
    
    # Step 2: Policy Training
    print("\n" + "=" * 80)
    print("Step 2: Policy Training (ULTIMATE with PPO – CLEAN VERSION)")
    print("=" * 80)
    print("Optimizations:")
    print("  - Algorithm: PPO (ổn định)")
    print("  - Learning rate: 1e-3")
    print("  - num_iter: 5 (demo, có thể tăng lên 150)")
    print("  - batch_size: 16")
    print("  - Baseline: linear")
    print("  - Gradient clipping: 0.5")
    print("  - Learning rate decay: 0.95")
    print("  - general policy enabled (for new clusters)")
    print("  - Transfer learning from DDQN enabled")
    print("=" * 80)
    

    
    policy_cmd = [
        sys.executable, "policy_training.py",
        "--sumo_config", sumo_config,
        "--model_path", model_path,
        "--output", output_path,
        "--algorithm", "ppo",
        "--opt", "adam",
        "--lr", "1e-3",
        "--num_iter", "10",
        "--num_periods", "5",
        "--device", "cuda",
        "--seed", "1009",
        "--batch_size", "16",
        "--hidden_size", "200",
        "--num_layers", "2",
        "--use_general_policy",
        "--max_steps", "7200",
        "--use_baseline",
        "--baseline", "linear",
        "--lr_decay", "0.95",
        "--lr_min", "1e-5",
        "--early_stop_patience", "10",
        "--early_stop_threshold", "0.01",
        "--grad_clip", "0.5",
        "--ddqn_init_path", "ddqn_sumo/output/120p4k/ddqn_model_final.pth"
    ]
    
    subprocess.run(policy_cmd, cwd=llirl_dir, check=True)
    
    print("\n" + "=" * 80)
    print("LLIRL ULTIMATE TRAINING COMPLETED (CLEAN VERSION – NO PERFORMANCE POLICY)!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Models: {llirl_dir}/{model_path}/")
    print(f"  - Output: {llirl_dir}/{output_path}/")
