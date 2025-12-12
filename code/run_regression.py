from pathlib import Path
from regression_utils import run_streaming_exact_ridge_for_subject


runid_list = ["sample_1e-5"]
for runid in runid_list:
    EMB_DIR=Path(f"./embeddings_{runid}")
    for subject_id in [2, 3]:
        SUBJECT_ID = subject_id 
        TRAIN_RATIO = 0.8
        ALPHAS = [1.0, 10.0, 100.0, 1000.0, 5000.0, 10000]
        #ALPHAS = [100.0]
        W, metrics = run_streaming_exact_ridge_for_subject(
            subject_id=SUBJECT_ID,
            emb_dir=EMB_DIR,
            train_ratio=TRAIN_RATIO,
            alphas=ALPHAS,
            K=3,
            compute_train_metrics=True,
            save_weights=True,
            seed=123,
            runid=runid,
        )
        #print(metrics)