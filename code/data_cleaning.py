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

def correct_typos(sentences):
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

    corrected_sentences = [
        [corrections.get(word, word) for word in sentence]
        for sentence in sentences
    ]
    return corrected_sentences


def remove_leading_and_trailing_spaces(sentences):
    """
    Removes leading and trailing spaces from strings in a list of lists.
    Input: [['  cat  ', 'dog  '], ['  bird']]
    Output: [['cat', 'dog'], ['bird']]
    """
    cleaned_data = [
        [word.strip() for word in sentence]
        for sentence in sentences
    ]
    return cleaned_data

def clean_nonword(sentences):
    """
    Removes specific unwanted tokens from a list of tokenized sentences.
    Input: list of lists (e.g., [['hello', 'world'], ['foo', 'bar']])
    """
    
    # define the blocklist as a set for faster lookup
    remove_set = { '', '  ', "'s", '(br}', '(br}','\\andm', '{cg}', '{ig}', '{ls)', '{ls} ', '{ns]', '{sl}',  'f{ns}'}

    # Use a nested list comprehension to filter the data
    # Logic: Keep the word ONLY IF it is NOT in the remove_set
    cleaned_data = [
        [word for word in sentence if word not in remove_set]
        for sentence in sentences
    ]

    return cleaned_data


def clean_data(pklfilename):
    """
    Data Cleaning Wrapper Function
    Input: pklfilename (raw_text.pkl)
    Output: [['cat', 'dog'], ['bird'], ...]
    """
    sentences = pkl2list(pklfilename) # convert pkl to list of lists
    sentences = correct_typos(sentences) # correct known typos
    sentences = remove_leading_and_trailing_spaces(sentences) # remove leading/trailing spaces
    sentences = clean_nonword(sentences) # remove non-words
    return sentences



