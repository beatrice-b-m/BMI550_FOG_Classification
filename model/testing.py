# import numpy as np
# import pandas as pd
from model.base import *

def train_model(model, train_df, test_df, target_feature, ngram_list, 
                n_features, vector_type, augment_dict, verbose: bool):
        # send descs to lists of desc object to store their attributes
        # and handle preprocessing/tokenizing
        train_descs = [FallDesc(
            i, 
            data[target_feature], 
            data.fall_description, 
            data.age_at_enrollment, 
            data.gender, 
            data.mds_updrs_iii_binary, 
            data.moca_total
        ) for i, data in train_df.iterrows()]
        test_descs = [FallDesc(
            i, 
            data[target_feature], 
            data.fall_description, 
            data.age_at_enrollment, 
            data.gender, 
            data.mds_updrs_iii_binary, 
            data.moca_total
        ) for i, data in test_df.iterrows()]

        # build train/val vocabulary
        train_desc_token_list = [d.ngram_list(ngram_list) for d in train_descs]
        # unpack token lists from each desc and take their set
        train_token_list = list(chain.from_iterable(train_desc_token_list))

        # get the set of unique tokens
        vocab_token_list = list(set(train_token_list))

        # if using tf-idf representation, build the document count dictionary
        if vector_type == 'tf-idf':
            doc_count_dict = build_doc_count_dict(train_desc_token_list, vocab_token_list)
        else:
            doc_count_dict = None

        # add all non-empty tokens to the vocab
        # we'll use an ordereddict so our vectors are consistent
        base_vocab_dict = OrderedDict((token, 0) for token in vocab_token_list if token)

        # build a frequency distribution from the training tokens
        freq_dist = nltk.FreqDist(train_token_list)
        # get the n most common tokens
        selected_token_list = [token for token, _ in freq_dist.most_common(n_features)]
        # filter the base_vocab dict to only contain the selected tokens
        base_vocab_dict = {k:v for k, v in base_vocab_dict.items() if k in selected_token_list}

        # build train/test x and y vectors
        train_x, train_y = get_x_y_vectors(
            train_descs, 
            ngram_list, 
            base_vocab_dict, 
            vector_type, 
            doc_count_dict, 
            augment_dict
        )
        test_x, test_y = get_x_y_vectors(
            test_descs, 
            ngram_list, 
            base_vocab_dict, 
            vector_type, 
            doc_count_dict, 
            augment_dict
        )

        # train and evaluate model
        eval_obj = fit_eval_model(model, train_x, train_y, test_x, test_y, return_metrics=True, verbose=verbose)
        return eval_obj