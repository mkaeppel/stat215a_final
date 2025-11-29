import sys
import os
import pickle
from gensim.models import Word2Vec
from data_cleaning import clean_data
import numpy as np 
from itertools import product
#from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction



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
# 1. Get the path to the folder ABOVE the 'code' folder
# (This assumes ridge_utils sits next to the 'code' folder)
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../final-project/code
parent_dir = os.path.dirname(current_dir)              # .../final-project

# Add that parent folder to the system path
sys.path.append(parent_dir)

# Load data
rawtext_fname = os.path.join(parent_dir, "data", "raw_text.pkl") # robust path building

with open(rawtext_fname, 'rb') as file:
    rawtext = pickle.load(file)


# # Generate the Word2Vec input from read data #
# sentences = []
# minword=1e10
# maxword=0
# for key, seq_obj in rawtext.items():
#     if hasattr(seq_obj, 'data'):
#         sentences.append(seq_obj.data)
        

#         if minword>len(seq_obj.data):
#             minword=len(seq_obj.data)
#         if maxword<len(seq_obj.data):
#             maxword = len(seq_obj.data)

# print(minword, maxword)
# print(f"Prepared {len(sentences)} documents/sentences for training.")

# ## Print the result ##

######################

sentences = clean_data(rawtext)
flattened_words = [item for sublist in sentences for item in sublist]
np.set_printoptions(threshold=sys.maxsize)
print( np.sort(np.unique(np.array(flattened_words))) )

### Train the model ###
model = Word2Vec(sentences=sentences, vector_size=100, window=10, min_count=10, workers=4, sg=1)
# I chose skip-gram since it  captures more nuanced semantic relationships (sg=1)
# vector_size: 100-300
# CBOW or skip-gram
# windows 10 for skipgram and 5 for CBOW
# sg = 0 for CBOW, and 1 for skip-gram



word_list = ['and', 'the', 'i', 'to', 'a', 'of', 'like', 'first']
for word in word_list:
    similar_words = model.wv.most_similar(word, )
    print("####")
    print(f"Words similar to '{word}': {similar_words}")
    print()
model.save("./model/word2vec.model")





#### PLOTTING ####

x_vals, y_vals, labels = reduce_dimensions(model)
x1_list = np.arange(-55, 50, 15)
y1_list = np.arange(-45, 45, 15)

y_desc = sorted(y1_list, reverse=True)  # largest y → smallest y

combinations = [(x, y) for y in y_desc for x in x1_list]


fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(x_vals, y_vals, s=5)
i_fig=1
for i, (x1, y1) in enumerate(combinations):
    x2 = x1 + 15
    y2 = y1 + 15
    mask = (x_vals < x2) & (x_vals > x1) & (y_vals < y2) & (y_vals > y1)
    idx = np.where(mask)[0]

    ax.axvline(x=x1, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=x2, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=y1, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=y2, color='gray', linestyle='--', linewidth=0.5)
    if len(idx)>0:
        ax.text(x1 + 1, y2 - 1, f"{i_fig}", color='k', fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='left')
        i_fig+=1

ax.set_xlim(-55, 50)
ax.set_ylim(-45, 45)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("Word2Vec Word Embeddings Visualized with t-SNE")
fig.tight_layout()
fig.savefig('./figures/word2vec_tsne_grid.png', dpi=600)
#plt.show()



i_fig=1
for i, (x1, y1) in enumerate(combinations):
    x2 = x1 + 15
    y2 = y1 + 15

    mask = (x_vals < x2) & (x_vals > x1) & (y_vals < y2) & (y_vals > y1)
    idx = np.where(mask)[0]
    if len(idx)>0:
        x_sel = x_vals[idx]
        y_sel = y_vals[idx]
        labels_sel = labels[idx]
        
        fig,ax = plt.subplots(figsize=(6,6))
        ax.scatter(x_sel, y_sel)

        for label, x, y in zip(labels_sel, x_sel, y_sel):
            ax.annotate(
                label,          # The text string
                xy=(x, y),      # The point (x, y) to annotate
                xytext=(2, 2),  # (Optional) Offset text slightly so it doesn't overlap the dot
                textcoords='offset points' 
            )
        ax.text(x1 + 0.1, y2 - 0.1, f"{i_fig}", color='k', fontsize=15, fontweight='bold',
            verticalalignment='top', horizontalalignment='left') 
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        fig.tight_layout()
        fig.savefig(f'./figures/word2vec_tsne_grid_{i_fig}.png', dpi=600)
        i_fig+=1

