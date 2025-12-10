import sys
import os
import pickle
from gensim.models import Word2Vec
from data_cleaning_pkl import clean_data
import numpy as np 
from itertools import product
#from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
from collections import Counter


def get_ordered_indices(labels, example_words):
    """
    Finds the indices of example_words within labels, 
    maintaining the specific order and frequency requested 
    in example_words.
    """
    ordered_idx = []
    # Create a mutable list/array of labels to mark items as "found"
    temp_labels = list(labels) 

    for word in example_words:
        try:
            # Find the index of the *next* occurrence of the word
            idx = temp_labels.index(word)
            ordered_idx.append(idx)
            
            # "Remove" the found element from the temporary list 
            # to ensure the next search for the same word finds 
            # the next unique index in the original array
            temp_labels[idx] = None 
        except ValueError:
            # Handle cases where a word in example_words is not in labels
            print(f"Warning: '{word}' not found in labels.")
            continue
            
    return np.array(ordered_idx)


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    #pca = IncrementalPCA(n_components=num_dimensions)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]

    return np.array(x_vals), np.array(y_vals), labels



"""
READ DATA
"""
##### raw data path #####
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project
sys.path.append(parent_dir)
rawtext_fname = os.path.join(parent_dir, "data", "raw_text.pkl") # robust path building
##################


#### Clean data ####
cleaned_data = clean_data(rawtext_fname)
#####################
# print(len(rawtext['sweetaspie'].data_times))
# print(len(rawtext['sweetaspie'].data))
# print(len(rawtext['sweetaspie'].tr_times))
# print(len(rawtext['sweetaspie'].split_inds))

### Train the model ###
from collections import Counter
import itertools

corpus = [ds.data for ds in cleaned_data.values()]  # List[List[str]]
all_words = list(itertools.chain.from_iterable(corpus))

# Keep only words with length > 5
long_words = [w for w in all_words if len(w) > 7]

# Count frequencies
counter = Counter(long_words)

# Sort by frequency (descending)
sorted_words = counter.most_common()   # list of (word, count) tuples

# For example, top 50
top_words = sorted_words


vector_size_list = [300] #, 50, 100, 200, 500] # the number of vec
sample_list = [1e-3] #range(10, 50, 5)


model_param_list = list(product(vector_size_list, sample_list))
model_list = []


tsne_output = np.load("./model/word2vec_tsne.npy")
x_vals = tsne_output[:,0].astype(float)
y_vals = tsne_output[:,1].astype(float)
labels = tsne_output[:,2]




example_words = [
    "discourse",     "disagreement",
    "thirteen",     "fourteen",
    #"students",     "colleagues",
    "theology",     "religious",
    "tolerance",     "thoughtful",
    "mother's",     "father's",
    "thursday",     "wednesday",
    #"paperwork",     "authorities",
    "defeating",     "obedient",
    "screaming",     "freaking",
    #"everyone's",     "everybody's",
    #"training",    "instructors",
    "intelligent",    "brilliant",
    "disoriented",    "reluctant",
    #"sometimes",    "probably",
    "bachelor's",    "curriculum",
    "messaged",    "linguistic",
    #"prosperous",    "cumulative",
    "anecdotes",    "entertain",
]
#mask = np.isin( )
idx = get_ordered_indices(labels, example_words)

x_labels = x_vals[idx]
y_labels = y_vals[idx]
lab_labels = labels[idx]


fig,ax = plt.subplots(figsize=(7,7))
ax.scatter(x_vals, y_vals, s=5, facecolor='gray', lw=1)
for i in range(len(example_words)):
    #perturb_x = np.random.uniform(low=-2, high=2)
    #perturb_y = np.random.uniform(low=-2, high=2)
    if i%2==0:
        ax.annotate(lab_labels[i],
                    xy = (x_labels[i], y_labels[i]),
                    xytext=(x_labels[i], y_labels[i]+4),
                    ha='center', fontweight='bold')
    else:
        ax.annotate(lab_labels[i],
                    xy = (x_labels[i], y_labels[i-1]),
                    xytext=(x_labels[i], y_labels[i-1]),
                    ha='center', fontweight='bold')
#ax.set_xlim(-100, -50)
#ax.set_ylim(-45, 30)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("Word2Vec Word Embeddings")
fig.tight_layout()
fig.savefig('./figures/word2vec_tsne_grid.png', dpi=600)
