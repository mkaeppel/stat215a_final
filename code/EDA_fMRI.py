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
for fmri_file in fmri_files[:1]:
    with open(fmri_file, 'rb') as file:
        fmri_data = np.load(file)
        df = pd.DataFrame(fmri_data[:,500:2500])
        corr_mat = df.corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr_mat, cmap='coolwarm', vmin=-0.7, vmax=0.7)
        fig.colorbar(cax)
        #fig, ax = plt.subplots()
        #implt = ax.imshow(fmri_data, aspect='auto', cmap='rainbow', vmin = -1, vmax=1)
        #fig.colorbar(implt, label='fMRI Signal Intensity', ax=ax)
        #plt.plot(range(len(fmri_data)), fmri_data[:,2])
        #plt.plot(range(len(fmri_data)), fmri_data[:,1])

plt.show()
        #print(np.shape(fmri_data))
        #pca = PCA(n_components=50)
        #pca.fit(fmri_data)



        #plt.plot(range(50),pca.explained_variance_ratio_)#print(pca.explained_variance_ratio_)
        #plt.yscale('log')
        #plt.show()
        #print(pca.explained_variance_ratio_.sum())
        
        #print(fmri_data)

        #n_scans = fmri_data['fmri_data'].shape[0]
        #n_scans_list.append(n_scans)
