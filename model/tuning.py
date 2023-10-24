import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import json
import logging
import random
from model.base import *

class HyperParamOptimizer:
    def __init__(self, model_function, model_name: str, hparam_range_dict: dict):
        # load model function and hyperparameter ranges
        self.model_function = model_function
        self.model_name = model_name
        self.hparam_range_dict = hparam_range_dict
        
        # initialize parameters
        self.train_val_df = None
        self.test_df = None
        self.target_feature = None
        self.vector_type = None
        self.n_features = None
        self.augment_dict = None
        self.seed = None
        self.k = None
        self.constant_dict = None
        
    def load_data(self, train_val_path, test_path):
        self.train_val_df = pd.read_csv(train_val_path)
        self.test_df = pd.read_csv(test_path)
        print("Data loaded...")
        
    def set_params(self, ngram_list: list or None = None, target_feature: str = 'fog_q_class', 
                   vector_type: str = 'tf-idf', n_features: int = 250, constant_dict: dict or None = None, 
                   augment_dict: dict or None = None, seed: int = 13, k: int = 5):
        # update parameters
        self.ngram_list = ngram_list
        self.target_feature = target_feature
        self.vector_type = vector_type
        self.n_features = n_features
        self.augment_dict = augment_dict
        self.seed = seed
        self.k = k
        self.constant_dict = constant_dict
        print("Parameters updated...")
        
    def index_conv_categorical_params(self, hparam_dict):
        """
        index converted categorical features back to their categorical values
        """
        out_hparam_dict = {}
        # iterate over variables and their ranges
        for k, v in hparam_dict.items():
            # if the feature is categorical
            if k[-4:] == "_CAT":
                old_k = k[:-4]
                # apply index to the list stored in old_k in the hparam range dict
                old_v = self.hparam_range_dict[old_k][int(v)]
                out_hparam_dict[old_k] = old_v
                
            else:
                out_hparam_dict[k] = v
                
        return out_hparam_dict
        
    def process_hparams(self):
        """
        function to convert categorical variables (stored as lists)
        to continuous ranges (stored as tuples)
        """
        out_hparam_dict = {}
        # iterate over variables and their ranges
        for k, v in self.hparam_range_dict.items():
            # if value is a list
            if isinstance(v, list):
                # convert key name to indicate it's categorical
                new_k = f"{k}_CAT"
                # set the non-inclusive max bound of the range as the length of the list
                max_bound = len(v) - 1e-7
                new_v = (0.0, max_bound)
                
                # register the new key and value range to the dictionary
                out_hparam_dict[new_k] = new_v
            else:
                out_hparam_dict[k] = v
                
        return out_hparam_dict
                
                
        
    def optimize(self, n_random, n_guided, opt_idx: str = '0'):
        # convert categorical hparams to continuous ranges
        hparam_dict = self.process_hparams()
        
        log_path = f"./logs/{self.model_name}_opt_{opt_idx}.json"
        
        # start the logger
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
        
        print(f"\nOptimizing {self.model_name} {'-'*60}")
        print(f"Logging results to '{log_path}'")

        # define optimizer
        optimizer = BayesianOptimization(
            f=self.objective_wrapper,
            pbounds=hparam_dict,
            random_state=self.seed)

        # maximize f1 with hyperparams
        optimizer.maximize(init_points=n_random, n_iter=n_guided)

        # open the log file
        opt_df = pd.read_json(log_path, lines=True)
        
        # extract the best performing parameters based on the micro-avg f1
        best_idx = opt_df['micro_avg_f1'].idxmax()
        best_param_dict = opt_df.loc[best_idx, 'params']
        
        # train a model and evaluate it on the test set with the best performing parameters
        print('Run using best params: ', best_param_dict)
        eval_obj = self.train_model(self.train_val_df, self.test_df, best_param_dict, verbose=False)
        
        # micro and macro average f1 are the same since we only have 1 test set/train set
        log_data(best_param_dict, eval_obj.acc, eval_obj.f1, eval_obj.f1, final=True)
        
    def objective_wrapper(self, **kwargs):
        """
        wrapper to pass the keyword arguments from the bayes opt package
        to the objective function as a dict since we're passing the params 
        to the model_function (not the objective function) as a dict.
        """
        hparam_dict = self.index_conv_categorical_params(kwargs)
        eval_metric = self.objective_function(hparam_dict)
        return eval_metric
        
    def objective_function(self, hparam_dict, sample_size: int or None = None, 
                           log: bool = True, return_all: bool = False, rfc: bool = False):
        """
        use val_df in place of test_df when optimizing hyperparams for models that
        don't take a validation set by default.
        """
        # if ngram list isn't defined, default to [1, 2]
        if self.ngram_list is None:
            self.ngram_list = [1, 2]

        # if augment dict isn't defined, default to False for all
        if self.augment_dict is None:
            self.augment_dict = {
                'token_length': False, 
                'patient_demographics': False, 
                'mds_updrs': False,
                'moca': False
            }

        # build an empty list to store the evaluation objects in
        eval_obj_list = []
        
        # if evaluating on the test set, randomly sample from the train set and test on the test set
        # if sample_size is -1, bootstrapping with k 60-100% length samples will be performed
        if sample_size is not None:
            print(f"Randomly sampling {self.k} sets with size {sample_size} from train set...")                    
            data_gen = rand_sample_gen(self.train_val_df, self.test_df, k=self.k, n=sample_size) 
            
        # otherwise perform k-fold cross-validation
        else:
            print(f"Splitting data into {self.k} folds...")
            data_gen = k_fold_generator(self.train_val_df, self.k)
        
        for train_df, test_df in data_gen:
            eval_obj = self.train_model(train_df, test_df, hparam_dict, verbose=False, rfc=rfc)
            
            # append the eval object to the list
            eval_obj_list.append(eval_obj)
            
        # get the macro/micro f1
        acc, micro_f1, macro_f1 = eval_optimizer_run(eval_obj_list)
        
        if log:
            log_data(hparam_dict, acc, micro_f1, macro_f1)
            
        if return_all:
            return acc, micro_f1, macro_f1
        else:
            return micro_f1
            
    def train_model(self, train_df, test_df, hparam_dict: dict, verbose: bool, rfc: bool):
        # send descs to lists of desc object to store their attributes
        # and handle preprocessing/tokenizing
        train_descs = [FallDesc(
            i, 
            data[self.target_feature], 
            data.fall_description, 
            data.age_at_enrollment, 
            data.gender, 
            data.mds_updrs_iii_binary, 
            data.moca_total
        ) for i, data in train_df.iterrows()]
        test_descs = [FallDesc(
            i, 
            data[self.target_feature], 
            data.fall_description, 
            data.age_at_enrollment, 
            data.gender, 
            data.mds_updrs_iii_binary, 
            data.moca_total
        ) for i, data in test_df.iterrows()]

        # build train/val vocabulary
        train_desc_token_list = [d.ngram_list(self.ngram_list) for d in train_descs]
        # unpack token lists from each desc and take their set
        train_token_list = list(chain.from_iterable(train_desc_token_list))

        # get the set of unique tokens
        vocab_token_list = list(set(train_token_list))

        # if using tf-idf representation, build the document count dictionary
        if self.vector_type == 'tf-idf':
            doc_count_dict = build_doc_count_dict(train_desc_token_list, vocab_token_list)
        else:
            doc_count_dict = None

        # add all non-empty tokens to the vocab
        # we'll use an ordereddict so our vectors are consistent
        base_vocab_dict = OrderedDict((token, 0) for token in vocab_token_list if token)

        # build a frequency distribution from the training tokens
        freq_dist = nltk.FreqDist(train_token_list)
        # get the n most common tokens
        selected_token_list = [token for token, _ in freq_dist.most_common(self.n_features)]
        # filter the base_vocab dict to only contain the selected tokens
        base_vocab_dict = {k:v for k, v in base_vocab_dict.items() if k in selected_token_list}

        # build train/test x and y vectors
        train_x, train_y = get_x_y_vectors(
            train_descs, 
            self.ngram_list, 
            base_vocab_dict, 
            self.vector_type, 
            doc_count_dict, 
            self.augment_dict
        )
        test_x, test_y = get_x_y_vectors(
            test_descs, 
            self.ngram_list, 
            base_vocab_dict, 
            self.vector_type, 
            doc_count_dict, 
            self.augment_dict
        )
        
        # if using an rfc with a smaller sample size, adjust the number of bins when necessary
        if rfc:
            train_len = len(train_df)
            n_bins = 128
            if train_len < n_bins:
                n_bins = train_len
            self.constant_dict['n_bins'] = n_bins
            
            
        
        # if the constant_dict is define, combine it with the hparam dict
        if self.constant_dict is not None:
            model_params = {**hparam_dict, **self.constant_dict}
        else:
            model_params = hparam_dict
        
        # build the model with the model function
        model = self.model_function(**model_params)

        # train and evaluate model
        eval_obj = fit_eval_model(model, train_x, train_y, test_x, test_y, return_metrics=True, verbose=verbose)
        return eval_obj
        
        
def eval_optimizer_run(eval_obj_list):
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    f1_list = []
    
    for eval_obj in eval_obj_list:
        total_tp += eval_obj.tp
        total_tn += eval_obj.tn
        total_fp += eval_obj.fp
        total_fn += eval_obj.fn
        
        f1_list.append(eval_obj.f1)
    
    # calculate the micro-average metrics
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    prec = total_tp / (total_tp + total_fp + 1e-6)
    rec = total_tp / (total_tp + total_fn + 1e-6)
    micro_f1 = (2 * prec * rec) / (prec + rec + 1e-6)
              
    # get the macro-average f1 by averaging the f1 from each run
    macro_f1 = sum(f1_list) / len(f1_list)
              
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Micro-Averaged F1: {micro_f1:.4f}")
    print(f"Macro-Averaged F1: {macro_f1:.4f}")
              
    return acc, micro_f1, macro_f1
                
def k_fold_generator(df, k, seed: int = 13):
    # shuffle the dataframe
    shuff_df = df.sample(frac=1, random_state=seed).reset_index()
    
    # split dataframe into a list of k folds of (near) equal size
    fold_list = np.array_split(shuff_df, k)
    
    # build a k length list of Trues
    base_mask = [True]*k
    
    # enumerate folds
    for i, fold in enumerate(fold_list):
        # copy base mask and set the ith index to False
        train_mask = base_mask.copy()
        train_mask[i] = False
        
        # set current chunk as the current validation df
        val_df = fold
        
        # apply the train mask to the chunk list and concatenate the
        # included folds
        train_df = pd.concat(
            [f for f, include in zip(fold_list, train_mask) if include], 
            axis=0
        )
        yield train_df, val_df
        
def rand_sample_gen(train_val_df, test_df, n, k):
    """
    instead of using a k-fold generator, yield a random (unseeded) sample from the train set
    k times.
    """
    for i in range(k):
        # if n is -1, randomly select a sample size between 70-100%
        if n == -1:
            sample_n = random.randint(int(0.7*len(train_val_df)), len(train_val_df))
        else:
            sample_n = n
                                             
        train_sample_df = train_val_df.sample(sample_n)
        yield train_sample_df, test_df

def log_data(param_dict, accuracy, micro_f1, macro_f1, final: bool = False):
    # convert the data to a dict
    log_dict = {
        "params": param_dict,
        "accuracy": accuracy,
        "micro_avg_f1": micro_f1,
        "macro_avg_f1": macro_f1,
        "final": final
    }
    
    # convert the dict to a json string
    json_str = json.dumps(log_dict)
    
    # log the json string
    logging.info(json_str)
    print("Results logged...")