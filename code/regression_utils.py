import numpy as np
from pathlib import Path
import pickle


def safe_std(var, eps=1e-8):
    """
    Convert a variance vector to std, guarding against zeros.
    Any variance < eps is clipped to eps to avoid division by zero.
    """
    var_clipped = np.maximum(var, eps)
    return np.sqrt(var_clipped)

def story_level_split(train_ratio, subject_id, emb_dir, seed=123):
    """
    Perform an 80/20 (or any ratio) train/test split at the STORY level.

    This ensures:
      - Streaming is possible
      - No memory explosion
      - No TR from the same story is split across train/test

    Args:
        train_ratio: float in (0,1), e.g. 0.8 for 80%
        subject_id: int
        emb_dir: Path to directory with preprocessed story files
        seed: random seed for reproducibility

    Returns:
        train_files: list[Path]
        test_files:  list[Path]
    """

    # Find all story files
    pattern = f"subject{subject_id}_*_Xdelayed.pkl"
    all_files = list(emb_dir.glob(pattern))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found for subject {subject_id}")

    # Shuffle deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(all_files)

    # Compute number of TRs per file to choose story split
    file_lengths = []
    for f in all_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        file_lengths.append(d["X_delayed"].shape[0])

    total_TRs = sum(file_lengths)
    target_train_TRs = train_ratio * total_TRs

    train_files = []
    test_files = []

    running_total = 0
    for f, nTR in zip(all_files, file_lengths):
        if running_total < target_train_TRs:
            train_files.append(f)
            running_total += nTR
        else:
            test_files.append(f)

    
    if len(train_files) == 0:
        train_files.append(test_files.pop())
    if len(test_files) == 0:
        test_files.append(train_files.pop())

    print(f"[SPLIT] Total stories = {len(all_files)}")
    print(f"[SPLIT] Total TRs = {total_TRs}")
    print(f"[SPLIT] Train TRs = {running_total} ({running_total/total_TRs:.2%})")
    print(f"[SPLIT] #Train stories = {len(train_files)}")
    print(f"[SPLIT] #Test stories  = {len(test_files)}")

    return train_files, test_files


def compute_train_stats(train_files):
    """
    First streaming pass over TRAIN stories to compute:

        - Feature means and stds over time (mean_X, std_X) for X_delayed
        - Voxel means and stds over time (mean_Y, std_Y) for bold

    Done via running sums/sums-of-squares (no large matrices stored).

    Returns:
        mean_X: (D,) float32
        std_X:  (D,) float32
        mean_Y: (V,) float32
        std_Y:  (V,) float32
        n_total: int, total number of TRAIN TRs
    """
    sum_X = None
    sumsq_X = None
    sum_Y = None
    sumsq_Y = None
    n_total = 0

    for f in train_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)

        X = d["X_delayed"].astype(np.float64)  # (N_i, D)
        Y = d["bold"].astype(np.float64)       # (N_i, V)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        N_i, D = X.shape
        Ny, V = Y.shape
        assert N_i == Ny, f"TR mismatch in {f}"

        if sum_X is None:
            sum_X = np.zeros(D, dtype=np.float64)
            sumsq_X = np.zeros(D, dtype=np.float64)
            sum_Y = np.zeros(V, dtype=np.float64)
            sumsq_Y = np.zeros(V, dtype=np.float64)

        sum_X += X.sum(axis=0)
        sumsq_X += (X ** 2).sum(axis=0)
        sum_Y += Y.sum(axis=0)
        sumsq_Y += (Y ** 2).sum(axis=0)

        n_total += N_i
        print(f"[PASS1] {f.name}: N_i={N_i}, running N={n_total}")

    mean_X = sum_X / n_total
    mean_Y = sum_Y / n_total

    var_X = sumsq_X / n_total - mean_X ** 2
    var_Y = sumsq_Y / n_total - mean_Y ** 2

    std_X = safe_std(var_X).astype(np.float32)
    std_X[std_X < 1e-6] = 1e-6 
    std_Y = safe_std(var_Y).astype(np.float32)
    std_Y[std_Y < 1e-6] = 1e-6

    mean_X = mean_X.astype(np.float32)
    mean_Y = mean_Y.astype(np.float32)

    print(f"[PASS1] Total TRAIN TRs: {n_total}")
    print(f"[PASS1] Feature dim D: {mean_X.shape[0]}, voxel dim V: {mean_Y.shape[0]}")

    return mean_X, std_X, mean_Y, std_Y, n_total


def accumulate_xtx_xty(train_files, mean_X, std_X, mean_Y, std_Y):
    """
    Second streaming pass over TRAIN stories to accumulate:

        XtX = sum(X_z^T X_z)  over all TRAIN TRs
        XtY = sum(X_z^T Y_z)  over all TRAIN TRs

    where:
        X_z = (X - mean_X) / std_X
        Y_z = (Y - mean_Y) / std_Y

    These are the sufficient statistics for exact ridge:

        (XtX + alpha I) W = XtY

    Returns:
        XtX: (D, D) float64
        XtY: (D, V) float64
    """
    D = mean_X.shape[0]
    V = mean_Y.shape[0]

    XtX = np.zeros((D, D), dtype=np.float64)
    XtY = np.zeros((D, V), dtype=np.float64)

    for f in train_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)

        X = d["X_delayed"].astype(np.float32)  # (N_i, D)
        Y = d["bold"].astype(np.float32)       # (N_i, V)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        Xz = (X - mean_X[None, :]) / std_X[None, :]
        Yz = (Y - mean_Y[None, :]) / std_Y[None, :]

        XtX += Xz.T @ Xz
        XtY += Xz.T @ Yz

        print(f"[PASS2] Accumulated XtX/XtY from {f.name}, N_i={X.shape[0]}")

    print(f"[PASS2] XtX shape: {XtX.shape}, XtY shape: {XtY.shape}")
    return XtX, XtY


def solve_ridge(XtX, XtY, alpha, eps=1e-8):
    """
    Stable ridge solver using eigendecomposition of XtX:

        XtX = Q diag(lambda) Q^T
        (XtX + alpha I)^(-1) XtY = Q diag(1/(lambda + alpha)) Q^T XtY
    """
    D = XtX.shape[0]
    print(f"[RIDGE] Solving ridge with D={D}, alpha={alpha} ...")

    # Symmetric PSD eigendecomposition
    eigvals, Q = np.linalg.eigh(XtX)   # XtX is symmetric

    # Regularized inverse eigenvalues
    denom = eigvals + alpha
    denom[denom < eps] = eps          # avoid division by 0 / tiny
    inv_diag = 1.0 / denom            # (D,)

    # Compute W = Q diag(inv_diag) Q^T XtY
    Qt_XtY = Q.T @ XtY                # (D, V)
    W = Q @ (inv_diag[:, None] * Qt_XtY)  # (D, V)

    W = W.astype(np.float32)

    # Sanity check for NaNs/Infs
    if not np.isfinite(W).all():
        n_nan = np.isnan(W).sum()
        n_inf = np.isinf(W).sum()
        raise ValueError(f"[RIDGE] Non-finite entries in W: NaN={n_nan}, Inf={n_inf}")

    print(f"[RIDGE] Done. W shape: {W.shape}")
    return W

# Not used in our implementation: memory-efficient SVD ridge solver
def solve_ridge_svd_chunked(train_files, mean_X, std_X, mean_Y, std_Y,
                             alpha, voxel_chunk=1000):
    """
    Memory-efficient Huth-style SVD ridge solver.

    Does a single SVD on the (train) stimulus matrix Xz, then solves ridge
    in voxel chunks to avoid holding full Yz in RAM.

    Parameters
    ----------
    train_files : list[Path]
    mean_X, std_X : feature normalization stats
    mean_Y, std_Y : voxel normalization stats
    alpha : scalar ridge penalty
    voxel_chunk : number of voxels to solve at a time (e.g., 500–3000)

    Returns
    -------
    W : (D, V) ridge weights
    """

    # 1) build Xz only (stimulus matrix)
    print("[RIDGE-SVD] Building Xz...")

    # Determine total TR count T and dims
    T = 0
    D = len(mean_X)
    V = len(mean_Y)

    for f in train_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        T += d["X_delayed"].shape[0]

    # Allocate Xz (this is the biggest matrix we store)
    Xz = np.zeros((T, D), dtype=np.float32)

    # Fill Xz
    offset = 0
    for f in train_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        X = np.nan_to_num(d["X_delayed"].astype(np.float32))
        Xi = (X - mean_X) / std_X
        Ni = Xi.shape[0]

        Xz[offset:offset+Ni] = Xi
        offset += Ni

    print(f"[RIDGE-SVD] Xz shape: {Xz.shape}  (T={T}, D={D})")

    # 2) compute SVD of Xz
    print("[RIDGE-SVD] Computing SVD...")
    U, S, Vh = np.linalg.svd(Xz, full_matrices=False)
    # Shapes:
    #   U:  (T, D)
    #   S:  (D,)
    #   Vh: (D, D)

    # Ridge shrinkage term
    shrink = S / (S**2 + alpha)

    # Precompute U^T for efficiency
    Ut = U.T  # shape (D, T)

    # Prepare output weight matrix
    W = np.zeros((D, V), dtype=np.float32)

    # 3) solve ridge in voxel chunks
    print(f"[RIDGE-SVD] Solving ridge in chunks of {voxel_chunk} voxels...")

    start_vox = 0
    while start_vox < V:
        end_vox = min(start_vox + voxel_chunk, V)
        chunk_size = end_vox - start_vox

        print(f"[RIDGE-SVD] Processing voxels {start_vox}–{end_vox} ...")

        # Build Yz chunk ------------------
        Yz = np.zeros((T, chunk_size), dtype=np.float32)
        offset = 0
        for f in train_files:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            Y = np.nan_to_num(d["bold"].astype(np.float32))
            Yi = (Y - mean_Y) / std_Y

            Ni = Yi.shape[0]
            # Extract the voxel slice
            Yz[offset:offset+Ni] = Yi[:, start_vox:end_vox]
            offset += Ni

        # Solve ridge for chunk ------------------
        # UR = U^T Y
        UR = Ut @ Yz  # shape (D, chunk_size)

        # W_chunk = Vh.T * shrink[:,None] @ UR
        W_chunk = (Vh.T * shrink[:, None]) @ UR

        # Store into full W
        W[:, start_vox:end_vox] = W_chunk.astype(np.float32)

        start_vox = end_vox

    print("[RIDGE-SVD] Done solving SVD-based ridge.")

    return W



def streaming_metrics(files, mean_X, std_X, mean_Y, std_Y, W, label="TRAIN"):
    """
    Compute MSE and voxelwise correlation for a set of story files (TRAIN or TEST),
    in a streaming fashion.

    We z-score X and Y with TRAIN stats, compute predictions:

        pred = X_z @ W

    and accumulate:
        - SSE for MSE
        - sufficient stats for correlation

    Inputs:
        files: list[Path] of story files (TRAIN or TEST)
        mean_X, std_X, mean_Y, std_Y: TRAIN stats
        W: (D, V)
        label: str, for logging ("TRAIN" or "TEST")

    Returns:
        mse: float
        corr: (V,) voxelwise correlation array
    """
    V = W.shape[1]

    sum_pred = np.zeros(V, dtype=np.float64)
    sum_resp = np.zeros(V, dtype=np.float64)
    sum_pred2 = np.zeros(V, dtype=np.float64)
    sum_resp2 = np.zeros(V, dtype=np.float64)
    sum_prod = np.zeros(V, dtype=np.float64)
    n_total = 0
    sse = 0.0

    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)

        X = d["X_delayed"].astype(np.float32)  # (N_i, D)
        Y = d["bold"].astype(np.float32)       # (N_i, V)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        Xz = (X - mean_X[None, :]) / std_X[None, :]
        Yz = (Y - mean_Y[None, :]) / std_Y[None, :]

        pred = Xz @ W  # (N_i, V)
        if not np.isfinite(pred).all():
            raise ValueError(f"[METRICS {label}] non-finite pred values in file {f.name}")
        if not np.isfinite(Yz).all():
            raise ValueError(f"[METRICS {label}] non-finite Yz values in file {f.name}")

        # SSE for MSE
        sse += np.sum((pred - Yz) ** 2)

        # Correlation stats
        sum_pred += pred.sum(axis=0)
        sum_resp += Yz.sum(axis=0)
        sum_pred2 += (pred ** 2).sum(axis=0)
        sum_resp2 += (Yz ** 2).sum(axis=0)
        sum_prod += (pred * Yz).sum(axis=0)
        n_total += X.shape[0]

        print(f"[METRICS {label}] {f.name}: N_i={X.shape[0]}, running N={n_total}")

    mse = sse / n_total

    # Compute voxelwise corr from accumulated stats
    mean_pred = sum_pred / n_total
    mean_resp = sum_resp / n_total

    E_pred2 = sum_pred2 / n_total
    E_resp2 = sum_resp2 / n_total
    E_prod = sum_prod / n_total

    var_pred = E_pred2 - mean_pred ** 2
    var_resp = E_resp2 - mean_resp ** 2
    cov = E_prod - mean_pred * mean_resp

    var_pred = np.maximum(var_pred, 1e-12)
    var_resp = np.maximum(var_resp, 1e-12)

    corr = cov / np.sqrt(var_pred * var_resp)
    mean_corr = float(np.mean(corr))

    print(f"[RESULT {label}] MSE={mse:.4f}, mean corr={mean_corr:.4f}")
    return float(mse), corr


def kfold_split(files, K, seed=123):
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    folds = []
    fold_size = int(np.ceil(len(files) / K))
    for i in range(K):
        fold = files[i*fold_size:(i+1)*fold_size]
        if len(fold) > 0:
            folds.append(fold)
    return folds


def cross_validate_alpha(train_files, alphas, K=5, seed=123):
    """
    Story-level K-fold cross-validation over alpha values.

    For each alpha:
      For each fold:
        train on (TRAIN - fold)
        validate on fold
      Compute mean val MSE across folds.

    Return best_alpha, cv_results
    """
    folds = kfold_split(train_files, K=K, seed=seed)

    print(f"[CV] Running {K}-fold CV over alphas: {alphas}")

    cv_results = {}

    for alpha in alphas:
        fold_mses = []

        for k, val_files in enumerate(folds):
            train_fold = [f for f in train_files if f not in val_files]

            # PASS 1 on fold-training
            mean_X, std_X, mean_Y, std_Y, _ = compute_train_stats(train_fold)

            # PASS 2 accumulate XtX, XtY
            XtX, XtY = accumulate_xtx_xty(train_fold, mean_X, std_X, mean_Y, std_Y)

            # Solve ridge
            W = solve_ridge(XtX, XtY, alpha)
            # W = solve_ridge_svd_chunked(
            #                             train_files=train_files,
            #                             mean_X=mean_X,
            #                             std_X=std_X,
            #                             mean_Y=mean_Y,
            #                             std_Y=std_Y,
            #                             alpha=alpha,
            #                             voxel_chunk=1000,   # tune this depending on RAM
            #                     )


            # Validation metrics
            val_mse, _ = streaming_metrics(val_files, mean_X, std_X, mean_Y, std_Y, W)
            fold_mses.append(val_mse)

            print(f"[CV] alpha={alpha} fold={k} val_mse={val_mse:.4f}")

        mean_mse = np.mean(fold_mses)
        cv_results[alpha] = mean_mse
        print(f"[CV] alpha={alpha} mean_val_mse={mean_mse:.4f}")

    best_alpha = min(cv_results, key=cv_results.get)
    print(f"[CV] Best alpha={best_alpha} (lowest mean validation MSE)")

    return best_alpha, cv_results

def run_streaming_exact_ridge_for_subject(
    subject_id,
    emb_dir,
    train_ratio=0.8,
    alphas=(1.0, 10.0, 100.0, 1000.0),
    K=5,
    compute_train_metrics=True,
    save_weights=True,
    seed=123,
):
    """
    End-to-end pipeline:
      1. Story-level 80/20 split
      2. Alpha CV (story-level K-fold)
      3. Retrain on all 80% TRAIN stories with best alpha
      4. Test evaluation on 20% hold-out

    Args:
        subject_id : int
        emb_dir : Path
        train_ratio : float, e.g. 0.8 for 80% of TRs in TRAIN (story-level)
        alpha : float, ridge penalty
        compute_train_metrics : bool
        save_weights : bool
        seed : int, for reproducible story shuffle

    Returns:
        W: (D, V) float32
        metrics: dict containing train/test MSE and corr
    """
    # 1) Story-level 80/20 split
    train_files, test_files = story_level_split(
        train_ratio=train_ratio,
        subject_id=subject_id,
        emb_dir=emb_dir,
        seed=seed,
    )

    # 2) Cross-validation to choose alpha
    best_alpha, cv_results = cross_validate_alpha(train_files, alphas, K=K, seed=seed)

    # 3) Retrain final model on ALL training stories with best_alpha
    print(f"[FINAL TRAIN] Using best alpha={best_alpha}")
    mean_X, std_X, mean_Y, std_Y, _ = compute_train_stats(train_files)
    XtX, XtY = accumulate_xtx_xty(train_files, mean_X, std_X, mean_Y, std_Y)
    W = solve_ridge(XtX, XtY, best_alpha)
    # W = solve_ridge_svd_chunked(
    #                                 train_files=train_files,
    #                                 mean_X=mean_X,
    #                                 std_X=std_X,
    #                                 mean_Y=mean_Y,
    #                                 std_Y=std_Y,
    #                                 alpha=best_alpha,
    #                                 voxel_chunk=1000,   # tune this depending on RAM
    #                         )



    metrics = {"cv_results": cv_results, "best_alpha": best_alpha}

    # Train metrics
    if compute_train_metrics:
        train_mse, train_corr = streaming_metrics(train_files, mean_X, std_X, mean_Y, std_Y, W)
        metrics["train_mse"] = train_mse
        metrics["train_corr"] = train_corr
        metrics["mean_train_corr"] = float(np.mean(train_corr))

    # Test metrics
    test_mse, test_corr = streaming_metrics(
        test_files, mean_X, std_X, mean_Y, std_Y, W, label="TEST"
    )
    metrics["test_mse"] = test_mse
    metrics["test_corr"] = test_corr
    metrics["mean_test_corr"] = float(np.mean(test_corr))

    # Save weights & metadata
    if save_weights:
        out_path = Path(f"/ocean/projects/mth250011p/smazioud/ridge_no_clean/subject{subject_id}ridge_streaming_80_20_k_3.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({
                "W": W.astype(np.float32),
                "best_alpha": best_alpha,
                "alphas": alphas,
                "cv_results": cv_results,
                "metrics": metrics,
                "mean_X": mean_X,
                "std_X": std_X,
                "mean_Y": mean_Y,
                "std_Y": std_Y,
                "train_files": [str(p) for p in train_files],
                "test_files": [str(p) for p in test_files],
            }, f)
        print(f"[SAVE] Saved weights + metadata to: {out_path}")

    return W, metrics