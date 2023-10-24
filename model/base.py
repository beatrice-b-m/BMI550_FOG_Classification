import pandas as pd
import numpy as np
import nltk as nltk
from nltk.stem import PorterStemmer
import os
import re
from collections import OrderedDict
from itertools import chain
from dataclasses import dataclass # , field

@dataclass
class FallDesc:
    """
    this is a class to hold the attributes of a fall description
    """
    index: str
    label: str
    text: str
    patient_age: int
    patient_gender: str
    mds_updrs_binary: str
    moca: int
        
    def token_list(self):
        """
        convert text to a list of tokens with the preprocess_tokenize function
        """
        return preprocess_tokenize(self.text)
    
    def ngram_list(self, n_list):
        """
        preprocess/tokenize text then generate ngrams for it
        """
        ngram_list = []
        for n in n_list:
            ngram_list += list(nltk.ngrams(self.token_list(), n))
            
        return ngram_list
    
    def token_list_len(self):
        """
        return the length of the token list
        """
        return len(self.token_list())
    
    def augment_vector(self, augment_dict):
        augment_list = []

        if augment_dict['token_length']:
            augment_list.append(self.token_list_len())

        if augment_dict['patient_demographics']:
            if self.patient_gender == 'Female':
                gender = 0.0
            else:
                gender = 1.0
            augment_list += [self.patient_age / 100, gender]

        if augment_dict['mds_updrs']:
            if self.mds_updrs_binary == 'mild':
                mds_updrs = 0.0
            else:
                mds_updrs = 1.0
            augment_list.append(mds_updrs)

        if augment_dict['moca']:
            augment_list.append(self.moca / 50)
            
        return augment_list            
    
    def build_vector(self, base_dict, ngram_list, vtype, doc_count_dict, augment_dict, eps: float = 1e-6):
        # make a copy of the base dict
        vec_dict = base_dict.copy()
        
        # get the list of ngrams for the vector
        vec_ngram_list = self.ngram_list(ngram_list)
        
        # if the vector type is count add occurences
        # otherwise set occurences to 1
        if (vtype == 'count') | (vtype == 'tf-idf'):
            dict_func = self.add_val
        else:
            dict_func = self.set_val
            vec_ngram_list = list(set(vec_ngram_list))
        
        # iterate over the set of the ngrams
        for ngram in vec_ngram_list:
                # apply the dict function to the dictionary
                if vec_dict.get(ngram) is not None:
                        vec_dict = dict_func(vec_dict, ngram)
                        
        # if vector type is tf-idf, divide the count vector by the # doc occurrences
        if vtype == 'tf-idf':
            out_dict = OrderedDict(
                (ngram, count/(doc_count_dict[ngram] + eps)) for ngram, count in vec_dict.items()
            )
            
        else:
            out_dict = vec_dict
        
        out_list = list(out_dict.values())
        
        if any(augment_dict.values()):
            out_list += self.augment_vector(augment_dict)
        
        return np.array(out_list)
    
    def add_val(self, sent_dict, ngram):
        sent_dict[ngram] += 1
        return sent_dict
        
    def set_val(self, sent_dict, ngram):
        sent_dict[ngram] = 1
        return sent_dict
    

def preprocess_tokenize(text):
    # ensure text is a string
    text = str(text)
    
    # lowercase text
    lower_text = text.lower()

    # remove punctuation
    stripped_text = strip_punctuation(lower_text)

    # tokenize text
    token_list = nltk.tokenize.word_tokenize(stripped_text)
    
    # remove stopwords
    english_stopwords = nltk.corpus.stopwords.words('english')
    token_list = [t for t in token_list if t not in english_stopwords]
    
    # stem tokens
    stemmer = PorterStemmer()
    token_list = [stemmer.stem(t) for t in token_list]

    return token_list

def strip_punctuation(target_str: str, punctuation_str: str = "][)(,'‘’.@1234567890:;%“”^_'|!#$&''`",
                      replace_list: list = ['-', '/']):
    """
    strip punctuation in punctuation_str from target_str and replace
    elements in replace_list with ' '
    reused this function in other hws
    """
    out_str = target_str.translate({ord(c): None for c in punctuation_str})
    for s in ["\n"] + replace_list:
        out_str = out_str.replace(s, " ")
#     out_str = remove_emojis(out_str)
    return out_str

def get_x_y_vectors(desc_list, ngram_list, base_vocab_dict, vector_type, doc_count_dict, augment_dict):
    # build x and y vectors
    x_vector_list = [d.build_vector(
        base_vocab_dict, 
        ngram_list, 
        vector_type, 
        doc_count_dict,
        augment_dict
    ) for d in desc_list]
    
#     if any(augment_dict.values()):
#         x_vector_list += augment_vector(desc_list, augment_dict)
    
    y_list = [d.label for d in desc_list]

    x_vector = np.array(x_vector_list).astype(np.float32)
    y_vector = np.array(y_list).astype(np.float32)
    
    return x_vector, y_vector
            
def get_metrics(pred, true, thresh: float = 0.5, verbose: bool = False):
    try:
        # count the number of tp, fp, fn, tn
        tp = len(np.where((pred == 1.) & (true == 1.))[0])
        fp = len(np.where((pred == 1.) & (true == 0.))[0])
        fn = len(np.where((pred == 0.) & (true == 1.))[0])
        tn = len(np.where((pred == 0.) & (true == 0.))[0])

        # calculate metrics
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = (2 * prec * rec) / (prec + rec + 1e-6)
        
    except ZeroDivisionError:
        pred[pred < thresh] = 0.0
        pred[pred >= thresh] = 1.0
        eval_obj = get_metrics(pred, true, thresh)
        return eval_obj
    
    if verbose:
        print(f"\nAccuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
    
    return Evaluation(acc, prec, rec, f1, tp, fp, fn, tn)

def fit_eval_model(model, train_x, train_y, test_x, test_y, return_metrics: bool = False, verbose: bool = False):
    # fit model
    model.fit(train_x, train_y)

    # evaluate model
    if verbose:
        print('\ntrain metrics:', '-'*30)
    _ = get_metrics(model.predict(train_x), train_y, verbose=verbose)

    if verbose:
        print('\ntest metrics:', '-'*30)
    eval_obj = get_metrics(model.predict(test_x), test_y, verbose=verbose)
    
    if return_metrics:
        return eval_obj
    
def build_doc_count_dict(nested_token_list, vocab_token_list):
    """
    function to count the number of documents the target tokens appear in
    """
    doc_count_dict = {}
    for token in vocab_token_list:
        doc_count_dict[token] = sum(token in sublist for sublist in nested_token_list)

    return doc_count_dict


@dataclass
class Evaluation:
    acc: float
    prec: float
    rec: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int