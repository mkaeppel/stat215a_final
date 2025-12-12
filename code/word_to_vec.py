import sys
import os
import pickle
from gensim.models import Word2Vec
from data_cleaning_pkl import clean_data
import numpy as np 
#from scipy.stats import norm
import matplotlib.pyplot as plt
#from sklearn.decomposition import IncrementalPCA    # inital reduction
#from sklearn.manifold import TSNE                   # final reduction
from preprocessing import downsample_word_vectors, make_delayed
from pathlib import Path
import pickle

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




OUT_DIR = Path("./embeddings_sample_1e-5")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = Path("./")
TEXT_PATH = BASE_DIR / "data" / "raw_text.pkl"

BOLD_BASE = BASE_DIR / "data"
SUBJECT_DIRS = {
    2: BOLD_BASE / "subject2",
    3: BOLD_BASE / "subject3",
}



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
corpus = [ds.data for ds in cleaned_data.values()]  # List[List[str]]

model = Word2Vec(
    sentences=corpus,
    vector_size=300,
    window=10,
    min_count=1,   # or >1 if you want to drop super-rare words
    workers=6,
    sg=1,           # skip-gram,
    sample=1e-3
)
model.save("./model/word2vec.model")


### T-sne ###
x_vals, y_vals, labels = reduce_dimensions(model)
tsne_output = np.c_[x_vals, y_vals, labels]
np.save("./model/word2vec_tsne.npy", tsne_output)

### Assign ###
word2vec_embeddings = dict()
for story_id, ds in cleaned_data.items():
    words = ds.data
    word_times = ds.data_times    # shape (T,)
    tr_times = ds.tr_times        # shape (n_TR,)

    # If some words might be OOV due to min_count > 1, you may want to filter or handle them
    # Here’s a simple version assuming all words are in the vocab (min_count=1):
    ordered_vectors = np.vstack([model.wv[w] for w in words])

    word2vec_embeddings[story_id] = {
        "words": words,
        "word_times": word_times,
        "tr_times": tr_times,
        "embeddings": ordered_vectors,
    }

wordseqs = cleaned_data


# downsample_word_vectors
stories = list(word2vec_embeddings.keys())
word_vectors = {}
for sid in stories:
    embs = word2vec_embeddings[sid]["embeddings"]   # (num_words, 768)
    word_vectors[sid] = embs.astype("float32")


for i, story in enumerate(stories):
    if i==0:
        print(wordseqs[story].__dict__.keys())
        print(len(wordseqs[story].split_inds))
        print(len(wordseqs[story].data))
        print(np.shape(wordseqs[story].data_times), np.shape(wordseqs[story].tr_times), np.shape(word_vectors[story]))


downsampled_semanticseqs = downsample_word_vectors(
    stories=stories,
    word_vectors=word_vectors,
    wordseqs=wordseqs
)

## UPDATED VERSION 12/08 ##
def preprocess_subject_streaming(subject_id, delays=None):
    subj_dir = SUBJECT_DIRS[subject_id]
    assert subj_dir.is_dir(), f"{subj_dir} does not exist"

    missing_stories = []

    for sid in stories:
        ds = wordseqs[sid]
        tr_times = ds.tr_times
        stim_tr  = downsampled_semanticseqs[sid]

        assert stim_tr.shape[0] == len(tr_times)

        bold_path = subj_dir / f"{sid}.npy"
        if not bold_path.is_file():
            print(f"[WARN] Subject {subject_id}: missing BOLD for story '{sid}', skipping.")
            missing_stories.append(sid)
            continue

        bold = np.load(bold_path)

        n_stim = stim_tr.shape[0]
        n_bold = bold.shape[0]

        if n_stim < n_bold:
            print(f"[WARN] {sid}: stim shorter than bold, skipping.")
            missing_stories.append(sid)
            continue

        trim_start = 10
        trim_end   = 5

        # check length
        if n_stim - trim_start - trim_end < n_bold:
            print(f"[WARN] {sid}: stim too short after fixed trimming, skipping.")
            missing_stories.append(sid)
            continue

        stim_trim = stim_tr[trim_start : n_stim - trim_end]

        # after trimming, must match bold length
        if stim_trim.shape[0] != n_bold:
            print(
                f"[WARN] {sid}: mismatch after fixed trim "
                f"(stim={stim_trim.shape[0]}, bold={n_bold}), skipping."
            )
            missing_stories.append(sid)
            continue

        if stim_trim.shape[0] != n_bold:
            print(f"[WARN] {sid}: mismatch after trim, skipping.")
            missing_stories.append(sid)
            continue

        # delay
        if delays is None:
            raise ValueError("delays must be provided if only X_delayed is saved.")

        X_delayed = make_delayed(stim_trim, delays=delays)
        X_delayed = X_delayed.astype("float32")

        bold = bold.astype("float32")

        result = {
            "X_delayed": X_delayed,   # (N, 768 * len(delays))
            "bold": bold,             # (N, n_vox)
        }

        out_file = OUT_DIR / f"subject{subject_id}_{sid}_Xdelayed.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(result, f)

        print(
            f"[SAVE] Subject {subject_id}, story {sid}: "
            f"X_delayed {X_delayed.shape}, bold {bold.shape}, saved"
        )

        del bold, stim_trim, X_delayed, result

    if missing_stories:
        print(f"\n[INFO] Subject {subject_id} skipped stories:")
        for s in missing_stories:
            print("  -", s)
    else:
        print(f"\n[INFO] Subject {subject_id}: all stories processed.")


        
delays = [1,2,3,4]
preprocess_subject_streaming(2, delays=delays)
preprocess_subject_streaming(3, delays=delays)


exit()