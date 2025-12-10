def summarize_ridge_results(res):
    metrics = res["metrics"]
    best_alpha = metrics["best_alpha"]
    cv_results = metrics["cv_results"]

    print("=== RIDGE SUMMARY ===")
    print(f"Best alpha: {best_alpha}")
    print("\n[CV results] mean validation MSE per alpha:")
    for a, mse in cv_results.items():
        print(f"  alpha={a:8g}  mean_val_MSE={mse:.4f}")

    # Train metrics (if present)
    if "train_mse" in metrics:
        print("\n[TRAIN]")
        print(f"  MSE          : {metrics['train_mse']:.4f}")
        print(f"  mean corr    : {metrics['mean_train_corr']:.4f}")

    # Test metrics
    print("\n[TEST]")
    print(f"  MSE          : {metrics['test_mse']:.4f}")
    print(f"  mean corr    : {metrics['mean_test_corr']:.4f}")

    print("\n[#voxel dimensions]")
    W = res["W"]
    print(f"  W shape (D, V): {W.shape}")


def plot_cv_results(res):
    cv_results = res["metrics"]["cv_results"]
    alphas = np.array(sorted(cv_results.keys()))
    mses = np.array([cv_results[a] for a in alphas])

    plt.figure(figsize=(4, 3))
    plt.plot(alphas, mses, marker="o")
    plt.xscale("log")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean validation MSE")
    plt.title("CV curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_corr_histograms(res, bins=50):
    metrics = res["metrics"]
    train_corr = metrics.get("train_corr", None)
    test_corr = metrics["test_corr"]

    plt.figure(figsize=(8, 3))

    # TRAIN
    if train_corr is not None:
        plt.subplot(1, 2, 1)
        plt.hist(train_corr, bins=bins)
        plt.axvline(metrics["mean_train_corr"], linestyle="--")
        plt.xlabel("correlation")
        plt.ylabel("#voxels")
        plt.title(f"TRAIN corr (mean={metrics['mean_train_corr']:.3f})")

    # TEST
    plt.subplot(1, 2, 2 if train_corr is not None else 1)
    plt.hist(test_corr, bins=bins)
    plt.axvline(metrics["mean_test_corr"], linestyle="--")
    plt.xlabel("correlation")
    plt.ylabel("#voxels")
    plt.title(f"TEST corr (mean={metrics['mean_test_corr']:.3f})")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

import pickle
outpklfname_s2 = "./regression_all/subject2ridge_streaming_80_20_k_3.pkl"
outpklfname_s3 = "./regression_all/subject3ridge_streaming_80_20_k_3.pkl"



with open(outpklfname_s2, 'rb') as file:
    res_s2 = pickle.load(file)

with open(outpklfname_s3, 'rb') as file:
    res_s3 = pickle.load(file)


def get_fft(pklresult):
    fs=1
    cc_voxel = pklresult['metrics']["test_corr"]
    cc_voxel = np.pad(cc_voxel, (0, 100000), mode='constant')
    X = np.fft.fft(cc_voxel)
    freqs = np.fft.fftfreq(len(cc_voxel), 1/fs)
    periods = 1/freqs
    amplitude = np.abs(X)
    return cc_voxel, periods, amplitude

cc_voxel_s2, periods_s2, amplitude_s2 = get_fft(res_s2)
cc_voxel_s3, periods_s3, amplitude_s3 = get_fft(res_s3)

from matplotlib.ticker import MultipleLocator

fig, ax = plt.subplots(2,1,figsize = (7,4))


ax[0].plot(cc_voxel_s2, color='#102522', lw=1, alpha=0.5, label="subject2")
ax[0].plot(cc_voxel_s3, color='#e8378b', lw=1, alpha=0.5, label="subject3")
ax[0].set_xlabel("voxel #")
ax[0].set_ylabel("CC")
ax[0].set_xlim(0,94000)
ax[0].xaxis.set_minor_locator(MultipleLocator(2000))
ax[0].grid(axis='x', which='major', alpha=0.3)
ax[0].grid(axis='x', which='minor', alpha=0.2)

ax[1].plot(periods_s2, amplitude_s2, color='#102522', alpha=0.5, label="subject2")
ax[1].plot(periods_s3, amplitude_s3, color='#e8378b', alpha=0.5, label="subject3")
ax[1].set_xlabel("Period")
ax[1].set_ylabel("Fourier Amplitude")
ax[1].set_xscale('log')
ax[1].legend()
#ax[1].set_yscale('log')
ax[1].set_xlim(10,10000)
ax[1].set_ylim(0,500)

## Find the maximum values ##
mask = np.where(  (periods_s2>1000) & (periods_s2<2000), True, False  )
print(np.c_[periods_s2[mask], amplitude_s2[mask]])
#print(amplitude[mask])


fig.tight_layout()
fig.savefig("./figures/cc_detailed.png", dpi=600)
plt.show()


# for i, key in enumerate(res):
#     print(key)['metrics']["test_corr"]
#     #print(res['metrics'])
#summarize_ridge_results(res)
#plot_cv_results(res)
#plot_corr_histograms(res)

# pre-trained model does not help
# 