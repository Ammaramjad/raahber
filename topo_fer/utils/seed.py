 from __future__ import annotations

 import os
 import random

 import numpy as np
 import torch
 from lightning import seed_everything


 def set_seed(seed: int, deterministic: bool = False) -> None:
     """Set global random seed for reproducibility."""
     seed_everything(seed, workers=True)
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     if deterministic:
         torch.use_deterministic_algorithms(True)
         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

