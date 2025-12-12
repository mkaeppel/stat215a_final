import sys
import os
import pickle
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import pandas as pd
import statsmodels.api as sm



###### LOAD RAW QUESTION DATA ####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
rawtext_fname = os.path.join(parent_dir, "data", "raw_text.pkl") # robust path building
with open(rawtext_fname, 'rb') as file:
    rawtext = pickle.load(file)
###################################

############ Basic EDA on text data ############
sentences = []
minword=1e10
maxword=0
Nwords_stories = []
Nuniquewords_stories = []
list_stories = []
for key, seq_obj in rawtext.items():
    #print(key, seq_obj)
    if hasattr(seq_obj, 'data'):
        sentences.append(seq_obj.data)

        if minword>len(seq_obj.data):
            minword=len(seq_obj.data)
        if maxword<len(seq_obj.data):
            maxword = len(seq_obj.data)
        Nwords_stories.append(len(seq_obj.data))      
        Nuniquewords_stories.append(len(np.unique(seq_obj.data)))
        list_stories.append(key)
    
print(np.mean(Nwords_stories), np.std(Nwords_stories), np.min(Nwords_stories), np.max(Nwords_stories))
print(np.mean(Nuniquewords_stories), np.std(Nuniquewords_stories), np.min(Nuniquewords_stories), np.max(Nuniquewords_stories))
fig, ax = plt.subplots(1,2, figsize=(7,3))
ax[0].hist(Nwords_stories, bins=30, color='grey', edgecolor='black')        
ax[1].hist(Nuniquewords_stories, bins=30, color='grey', edgecolor='black')
ax[0].set_xlabel("Number of words per story")
ax[1].set_xlabel("Number of unique words per story")
ax[0].set_ylabel("Number of stories")
ax[1].set_ylabel("Number of stories")
ax[0].set_xlim(500,3500)
ax[1].set_xlim(200,1100)
ax[0].set_ylim(0,12)
ax[1].set_ylim(0,18)
fig.tight_layout()
fig.savefig("./figures/word_counts_histogram.png", dpi=600)
##################################



#### WHOLE STORY STATISTICS ####
flattened_list = [item for sublist in sentences for item in sublist]
print("Total number of words of all 109 stories:", len(flattened_list))
print("Total number of UNIQUE words of all 109 stories:", len(np.unique(flattened_list)))

word_counts = collections.Counter(flattened_list)
top_100_frequent_words = word_counts.most_common(100)
df_top_100_frequent_words = pd.DataFrame(top_100_frequent_words, columns=['words', 'counts'])
print(df_top_100_frequent_words)
df = df_top_100_frequent_words.sort_values(by='counts', ascending=False)
fig, ax = plt.subplots(1,3, figsize=(7,2.5))
ax[0].barh(df['words'].iloc[:10], df['counts'].iloc[:10], color='teal')
ax[1].barh(df['words'].iloc[10:20], df['counts'].iloc[10:20], color='teal')
ax[2].barh(df['words'].iloc[20:30], df['counts'].iloc[20:30], color='teal')
ax[0].invert_yaxis()
ax[1].invert_yaxis()
ax[2].invert_yaxis()
ax[0].set_xlabel("counts")
ax[1].set_xlabel("counts")
ax[2].set_xlabel("counts")
ax[0].set_xlim(0, 10000)
ax[1].set_xlim(0, 10000)
ax[2].set_xlim(0, 10000)
fig.tight_layout()
fig.savefig("./figures/top_30_frequent_words.png", dpi=600)
######################################


###### LOAD FMRI DATA ####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
fmri_datapath = os.path.join(parent_dir, "data", "subject3") # robust path building
DATA_DIR = Path(fmri_datapath)
#############################


##### Number of fMRI observations per story #####
fig, ax = plt.subplots(figsize=(3,3))
list_Nobs = []
for i, datafile in enumerate(DATA_DIR.iterdir()):
    
    dat = np.load(datafile)
    storyname = datafile.stem    
    print(datafile.stem, np.min(dat), np.max(dat), np.shape(dat))
    Ntr = dat.shape[0]
    list_Nobs.append(Ntr)

ax.hist(list_Nobs)
ax.set_xlabel("# fMRi per story")
ax.set_ylabel("counts")

fig.tight_layout()
fig.savefig(f"./figures/Ntr_fmri.png", dpi=600)
##########################################


##### Q-Q PLOT of fMRI data #####
fig, ax = plt.subplots(1,2, figsize=(7,3))
list_Nobs = []
for i, datafile in enumerate(DATA_DIR.iterdir()):
    if i<2:
        dat = np.load(datafile)
        storyname = datafile.stem
        
        print(datafile.stem, np.min(dat), np.max(dat), np.shape(dat))
        #dat.flatten()
        sm.qqplot(dat.flatten(), alpha=0.4, line='s', ax=ax[i], markersize=3)
        #sm.qqplot(dat, alpha=0.1, line='s', ax=ax[1])
        ax[i].set_xlim(-6,6)
        ax[i].set_ylim(-6,6)
        ax[i].grid(which='both', linewidth=0.3)
fig.tight_layout()
fig.savefig(f"./figures/qqplot_fmri.png", dpi=600)
#############################################
