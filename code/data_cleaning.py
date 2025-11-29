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



def remove_leading_spaces(sentences):
    """
    Removes only leading spaces from strings in a list of lists.
    Input: [['  cat', 'dog'], ['  bird']]
    Output: [['cat', 'dog'], ['bird']]
    """
    cleaned_data = [
        [word.lstrip() for word in sentence] 
        for sentence in sentences
    ]
    return cleaned_data

def clean_nonword(sentences):
    """
    Removes specific unwanted tokens from a list of tokenized sentences.
    Input: list of lists (e.g., [['hello', 'world'], ['foo', 'bar']])
    """
    sentences = remove_leading_spaces(sentences)
    # define the blocklist as a set for faster lookup
    remove_set = { '',  "'s", '(br}', '(br}','\\andm', '{cg}', '{ig}', '{ls)', '{ls} ', '{ns]', '{sl}',  }

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
    Input: [['  cat', 'dog'], ['  bird']]
    Output: [['cat', 'dog'], ['bird']]
    """
    sentences = pkl2list(pklfilename)
    sentences = clean_nonword(sentences)
    return sentences


