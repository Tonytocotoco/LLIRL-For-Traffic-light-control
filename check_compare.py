import torch
import numpy as np
import torch.serialization

torch.serialization.add_safe_globals([np.ndarray, np.core.multiarray._reconstruct])

ckpt = torch.load(
    r"llirl_sumo/saves/120p4k_ultimate_test2/policies_final.pth",
    map_location="cpu",
    weights_only=False
)
print("Loaded OK!")
print("Keys:", ckpt.keys())
print("num_policies:", ckpt["num_policies"])
