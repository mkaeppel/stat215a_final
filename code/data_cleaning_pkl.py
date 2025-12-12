import pickle
import numpy as np
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

def correct_typos(pklfile):
    corrections = {
        "tennant": "tenant",
        "probablys": "probably",
        "fiften": "fifteen",
        "and'": "and",
        "secre": "secret",
        "intp": "into",
        "weaknesst": "weakest",
        "happene": "happen",
        "deppressing": "depressing",
        "whatw": "what",
        "that''s": "that's",
        "trow": "trou",
        "so]]": "so",
        "thikn": "think",
        "sid": "side",
        "bu": "but",
        "ofso": "of",
        "rea": "real",
        "walkd": "walked",
        "complulsively": "compulsively",
        "undertand": "understand",
        "welcone": "welcome",
        "abercormbie": "abercrombie",
        "temping": "tempting",
        "absinence": "abstinence",
        "adopteest": "adoptees",
        "inactivtiy": "inactivity",
        "onw": "one",
        "happeist": "happiest",
        "saety": "safety",
        "botter": "bottle",
        "chekcpoint": "checkpoint",
        "ao1": "all",
        "thity": "thirty",
        "successfull": "successful",
        "chaplainsometimes": "chaplain",
        "wasw": "was",
        "spanding": "spanning",
        "starte": "started",
        "did'nt": "didn't",
        "thier": "their",
    }
    for story in pklfile:
        #print(pklfile[story].data)
        corrected_data = [
            [corrections.get(word, word) for word in pklfile[story].data]
        ]
        pklfile[story].data = corrected_data[0]
    return pklfile


def remove_leading_and_trailing_spaces(pklfile):
    """
    Removes leading and trailing spaces from strings in a list of lists.
    Input: [['  cat  ', 'dog  '], ['  bird']]
    Output: [['cat', 'dog'], ['bird']]
    """
    for story in pklfile:
        clean_words = [w.strip() for w in pklfile[story].data]
        pklfile[story].data = clean_words
    return pklfile


def clean_nonword(pklfile):
    """
    Removes specific unwanted tokens from a list of tokenized sentences. Also, remove the corresponding data_times
    Input: list of lists (e.g., [['hello', 'world'], ['foo', 'bar']])
    """
    
    remove_set = { '', '  ', "'s", '(br}', '(br}','\\andm', '{cg}', '{ig}',
               '{ls)', '{ls} ', '{ns]', '{sl}', 'f{ns}' }
    
    for story in pklfile:
        w = pklfile[story].data
        t = pklfile[story].data_times
        tr = pklfile[story].tr_times
        if story == 'adollshouse':
            print(pklfile[story].data_times[:100])
            print(pklfile[story].split_inds)
            print(pklfile[story].tr_times)

        new_w = []
        new_t = []

        for word, time in zip(w, t):
            word_stripped = word.strip()  # remove leading/trailing spaces
            if word_stripped in remove_set:
                continue                  # skip this word & its time
            new_w.append(word_stripped)
            new_t.append(time)
        new_split_idx = np.searchsorted(new_t, tr, side="right")

        # write back
        pklfile[story].data = new_w
        pklfile[story].data_times = new_t

        print("****************")
        print(pklfile[story].split_inds)
        pklfile[story].split_inds = new_split_idx
        print(pklfile[story].split_inds)
        print("****************")

        
    return pklfile

def read_pkl(pklfilepath):
    with open(pklfilepath, 'rb') as file:
        rawtext = pickle.load(file)
    return rawtext


def clean_data(pklfilepath):
    """
    Data Cleaning Wrapper Function
    Input: pklfilename (raw_text.pkl)
    Output: [['cat', 'dog'], ['bird'], ...]
    """
    cleaned_pkl = read_pkl(pklfilepath)
    cleaned_pkl = correct_typos(cleaned_pkl) # correct known typos
    cleaned_pkl = remove_leading_and_trailing_spaces(cleaned_pkl) # remove leading/trailing spaces
    #cleaned_pkl = clean_nonword(cleaned_pkl) # remove non-words
    return cleaned_pkl

