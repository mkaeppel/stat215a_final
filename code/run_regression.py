from pathlib import Path
import numpy as np
from regression_utils import run_streaming_exact_ridge_for_subject

EMB_DIR=Path("/ocean/projects/mth250011p/smazioud/preprocessing/")

for subject_id in [3]:
    SUBJECT_ID = subject_id 
    TRAIN_RATIO = 0.8
    ALPHAS = np.logspace(1, 5, 20)

    W, metrics = run_streaming_exact_ridge_for_subject(
        subject_id=SUBJECT_ID,
        emb_dir=EMB_DIR,
        train_ratio=TRAIN_RATIO,
        alphas=ALPHAS,
        K=3,
        compute_train_metrics=True,
        save_weights=True,
        seed=123,
    )