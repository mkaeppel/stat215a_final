#!/bin/bash
conda activate 215a

# Run all scripts in order

#EDA
python3 EDA.py
python3 EDA_fMRI.py

# Embeddings
python3 word_to_vec.py
python3 word_to_vec_uncleaned.py
python3 word_to_vec_tsne.py
jupyter nbconvert --to notebook --execute --inplace GloVe.ipynb
jupyter nbconvert --to notebook --execute --inplace bert_pretrained.ipynb
jupyter nbconvert --to notebook --execute --inplace bert_finetuned.ipynb
jupyter nbconvert --to notebook --execute --inplace bert_finetuned_no_clean.ipynb

# Run regression for each embedding
python3 run_regression.py 

# Cross-validation
jupyter nbconvert --to notebook --execute --inplace cross_val_regression.ipynb
python3 detailed_cc_analysis.py 

# Interpreatation
jupyter nbconvert --to notebook --execute --inplace interpretation.ipynb

# Stability
jupyter nbconvert --to notebook --execute --inplace stability.ipynb