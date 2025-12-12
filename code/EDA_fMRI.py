import sys
import os
import pickle
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
story='A Doll\'s House'


###### LOAD FMRI DATA ####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
fmri_datapath = os.path.join(parent_dir, "data", "subject2") # robust path building
DATA_DIR = Path(fmri_datapath)
#############################


############ Basic EDA on fMRI data ############
fmri_files = list(DATA_DIR.glob("*.npy"))
n_scans_list = []
fig, axes = plt.subplots(1, 2, figsize=(7,3))
ax = axes[0]
for fmri_file in fmri_files[:1]:
    with open(fmri_file, 'rb') as file:
        fmri_data = np.load(file)
        
        df = pd.DataFrame(fmri_data[:,:5000])
        corr_mat = df.corr()

        cax = ax.matshow(corr_mat, cmap='seismic', vmin=-1, vmax=1)
        ax.text(0.01, 0.01, f"subject2, {story}", transform = ax.transAxes, ha='left', va='bottom')
        ax.set_ylabel("voxel #")
        ax.set_xlabel("voxel #")
        ax.xaxis.set_label_position("top")  # move label to top
        fig.colorbar(cax, shrink = 0.85)
        





###### LOAD FMRI DATA ####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
fmri_datapath = os.path.join(parent_dir, "data", "subject3") # robust path building
DATA_DIR = Path(fmri_datapath)
#############################


############ Basic EDA on fMRI data ############
fmri_files = list(DATA_DIR.glob("*.npy"))
n_scans_list = []
#fig, ax = plt.subplots()
ax = axes[1]
for fmri_file in fmri_files[:1]:
    with open(fmri_file, 'rb') as file:
        fmri_data = np.load(file)
        df = pd.DataFrame(fmri_data[:,:5000])
        corr_mat = df.corr()
        cax = ax.matshow(corr_mat, cmap='seismic', vmin=-1, vmax=1)
        #fig.colorbar(cax)
        ax.text(0.01, 0.01, f"subject3, {story}", transform = ax.transAxes, ha='left', va='bottom')
        ax.set_ylabel("voxel #")
        ax.set_xlabel("voxel #")
        ax.xaxis.set_label_position("top")  # move label to top
        fig.colorbar(cax, shrink = 0.85, label="CC over voxels")
        #fig, ax = plt.subplots()
        #implt = ax.imshow(fmri_data, aspect='auto', cmap='rainbow', vmin = -1, vmax=1)
        #fig.colorbar(implt, label='fMRI Signal Intensity', ax=ax)
        #plt.plot(range(len(fmri_data)), fmri_data[:,2])
        #plt.plot(range(len(fmri_data)), fmri_data[:,1])
fig.tight_layout()
fig.savefig("./figures/cc_voxel.png", dpi=600)



###### LOAD FMRI DATA ####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
fmri_datapath = os.path.join(parent_dir, "data", "subject2") # robust path building
DATA_DIR = Path(fmri_datapath)
#############################



############ Basic EDA on fMRI data ############
fmri_files = list(DATA_DIR.glob("*.npy"))
n_scans_list = []
fig, axes = plt.subplots(1, 2, figsize=(7,3))
ax = axes[0]
for fmri_file in fmri_files[:1]:
    with open(fmri_file, 'rb') as file:
        fmri_data = np.load(file)
        #df = pd.DataFrame(fmri_data[:,:1000])
        #corr_mat = df.corr()

        cax = ax.matshow(fmri_data, cmap='coolwarm', aspect=350, vmin=-2, vmax=2)
        ax.text(0.03, 0.03, f"subject2, {story}", transform = ax.transAxes, ha='left', va='bottom',
                bbox=dict(
                    facecolor="white",       # box fill color
                    edgecolor="black",       # box border color
                    boxstyle="round",        # 'round', 'square', 'round4', etc.
                    alpha=0.8                # transparency
    ))
        ax.set_ylabel("time stamp (TR)")
        ax.set_xlabel("voxel #")
        #ax.ticklabel_format(style="sci", axis="x")  # optional
        ax.xaxis.set_label_position("top")  # move label to top
        fig.colorbar(cax, shrink = 0.8)
        



###### LOAD FMRI DATA ####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
fmri_datapath = os.path.join(parent_dir, "data", "subject3") # robust path building
DATA_DIR = Path(fmri_datapath)
#############################



############ Basic EDA on fMRI data ############
fmri_files = list(DATA_DIR.glob("*.npy"))
n_scans_list = []
ax = axes[1]
for fmri_file in fmri_files[:1]:
    with open(fmri_file, 'rb') as file:
        fmri_data = np.load(file)
        #df = pd.DataFrame(fmri_data[:,:1000])
        #corr_mat = df.corr()

        cax = ax.matshow(fmri_data, cmap='coolwarm', aspect=350, vmin=-2, vmax=2)
        ax.text(0.03, 0.03, f"subject3, {story}", transform = ax.transAxes, ha='left', va='bottom',
                bbox=dict(
                    facecolor="white",       # box fill color
                    edgecolor="black",       # box border color
                    boxstyle="round",        # 'round', 'square', 'round4', etc.
                    alpha=0.8                # transparency
    ))
        #ax.set_ylabel("time stamp (TR)")
        ax.set_xlabel("voxel #")
        #ax.ticklabel_format(style="sci", axis="x")  # optional
        ax.xaxis.set_label_position("top")  # move label to top
        fig.colorbar(cax, shrink = 0.8, label="signal intensity")

fig.tight_layout()
fig.savefig("./figures/signal_voxel.png", dpi=600)
#fig.savefig()
########################################################




plt.show()