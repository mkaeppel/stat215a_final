#import nltk
#from nltk.corpus import wordnet
#from nltk.corpus import stopwords
#import sys
from collections import defaultdict # Useful for grouping occurrences
import nltk
from nltk.corpus import words, wordnet, stopwords
import numpy as np
#import sys
#from nltk.corpus import stopwords
import os
import pickle
import sys


# Generate the Word2Vec input from read data #
def pkl2list(pklfilename):
    """
    Read pkl file and convert to list of lists
    Input: pkl filename
    Output: list of lists
    """
    sentences = []
    for key, seq_obj in pklfilename.items():
        if hasattr(seq_obj, 'data'):
            sentences.append(seq_obj.data)
    return sentences


def check_spell_extensive_contextual_unique(sentences):
    """
    Checks the existence of words using an extensive combined list, and 
    presents the interactive review for UNIQUE possible typo words.
    
    Args:
        sentences (list): A nested list of words (e.g., [['word1', 'word2'], ['word3']]).
    """
    
    # --- 1. Download Required Resources ---
    nltk.download('wordnet')
    nltk.download('stopwords')
    
    # --- 2. Create the Extensive Word Set (Stopwords) ---
    english_stopwords_set = set(stopwords.words('english'))

    # --- 3. Flatten the sentences and Identify Typos ---
    all_words = []
    for sublist in sentences:
        all_words.extend(sublist)

    # Dictionary to store unique typo words and ALL their indices/occurrences:
    # { 'typo_word': [index1, index2, ...], ... }
    unique_typos_with_indices = defaultdict(list)
    
    for i, word in enumerate(all_words):
        word_lower = word.lower()
        
        # Check 1: Is it a common function word (via Stopwords)?
        is_stopword = word_lower in english_stopwords_set
        
        # Check 2: Does WordNet recognize it (via synsets)?
        is_wordnet_word = len(wordnet.synsets(word_lower)) > 0
        
        exists = is_stopword or is_wordnet_word
        
        if not exists:
            # Group by the word itself (case-sensitive)
            unique_typos_with_indices[word].append(i)
            
    # --- 4. Interactive One-by-One Review (Unique Words) ---
    print("\n--- Interactive Typo Review (Unique Words, ± 10 Words Context) ---")
    
    unique_typo_words = list(unique_typos_with_indices.keys())
    typo_count = len(unique_typo_words)
    
    if typo_count == 0:
        print("🎉 No potential typos found!")
        return True

    print(f"Found {typo_count} unique potential typos. Reviewing one by one...")

    # Iterate over the unique words
    for i, typo in enumerate(unique_typo_words):
        
        # Get all indices where this specific typo appears
        indices = unique_typos_with_indices[typo]
        
        # --- Print Review Card Header for the UNIQUE Word ---
        print("\n" + "=" * 60)
        print(f"UNIQUE Typo #{i + 1} of {typo_count}: **{typo}** (Appears {len(indices)} time{'s' if len(indices) > 1 else ''})")
        print("=" * 60)
        
        # Show all contexts for this unique word
        for occurrence_j, index in enumerate(indices):
            
            # Calculate the context window indices
            start_index = max(0, index - 10)
            end_index = min(len(all_words), index + 11)
            
            context_list = all_words[start_index:end_index]
            
            # Format the context string, highlighting the typo
            context_str_parts = []
            for w in context_list:
                if w == typo:
                    context_str_parts.append(f'**[{w}]**')
                else:
                    context_str_parts.append(w)

            context_str = ' '.join(context_str_parts)
            
            # --- Print Context for this Occurrence ---
            print(f"  Occurrence {occurrence_j + 1} (Index {index}):")
            print(f"    ... {context_str} ...")
            
        # Pause and wait for user input AFTER all contexts for the unique word are shown
        print("-" * 60)
        user_input = input("Press ENTER to review the next UNIQUE word, or type 'q' and ENTER to quit: ").strip().lower()
        
        if user_input == 'q':
            print("\nReview stopped by user.")
            break
            
    print("\n--- Interactive Review Complete ---")
    return True



def check_spell_extensive_contextual_unique_no_interaction(sentences):
    """
    Checks the existence of words using an extensive combined list, and 
    prints UNIQUE possible typo words in a 5-column format (no interaction).
    
    Args:
        sentences (list): A nested list of words 
                          (e.g., [['word1', 'word2'], ['word3']]).
    Returns:
        list: Sorted list of unique potential typo words.
    """
    
    # --- 1. Download Required Resources (only needed once per session) ---
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # --- 2. Create the Extensive Word Set (Stopwords) ---
    english_stopwords_set = set(stopwords.words('english'))

    # --- 3. Flatten the sentences and Identify Typos ---
    all_words = []
    for sublist in sentences:
        all_words.extend(sublist)

    # Dictionary to store unique typo words and ALL their indices/occurrences:
    # { 'typo_word': [index1, index2, ...], ... }
    unique_typos_with_indices = defaultdict(list)
    
    for i, word in enumerate(all_words):
        word_lower = word.lower()
        
        # Check 1: Is it a common function word (via Stopwords)?
        is_stopword = word_lower in english_stopwords_set
        
        # Check 2: Does WordNet recognize it (via synsets)?
        is_wordnet_word = len(wordnet.synsets(word_lower)) > 0
        
        exists = is_stopword or is_wordnet_word
        
        if not exists:
            unique_typos_with_indices[word].append(i)
    
    unique_typo_words = sorted(unique_typos_with_indices.keys())
    typo_count = len(unique_typo_words)
    
    print("\n--- Potential Unique Typos (5-column format) ---")
    if typo_count == 0:
        print("🎉 No potential typos found!")
        return []

    print(f"Found {typo_count} unique potential typos.\n")

    # --- 4. Print in 5-column format ---
    cols = 5
    col_width = 20  # adjust as you like

    for i, word in enumerate(unique_typo_words, start=1):
        print(f"{word:<{col_width}}", end="")
        if i % cols == 0:
            print()  # newline after each row of `cols` items

    # Final newline if last row was not complete
    if typo_count % cols != 0:
        print()

    return unique_typo_words



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




######################
#check_spell_extensive_contextual_unique(pkl2list(rawtext))
check_spell_extensive_contextual_unique_no_interaction(pkl2list(rawtext))
#sentences = clean_data(rawtext)

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