import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from cuml import SGD
from cuml.common.device_selection import set_global_device_type, get_global_device_type
from model.tuning import *

set_global_device_type('gpu')
print('new device type:', get_global_device_type())

# define model to optimize
model_function = SGD
model_name = "SGD"

# define hyperparameter ranges to search
hparam_range_dict = {
    'penalty': ['none', 'l1', 'l2', 'elasticnet'], 
    'eta0': (0.000001, 0.01),
    'l1_ratio': (0.001, 0.999)
}

# define optimizer
hparam_optimizer = HyperParamOptimizer(
    model_function=model_function, 
    model_name=model_name, 
    hparam_range_dict=hparam_range_dict
)

# load data into optimizer
train_val_path = "./data/fallreports_2023-9-21_train.csv"
test_path = "./data/fallreports_2023-9-21_test.csv"
hparam_optimizer.load_data(
    train_val_path=train_val_path, 
    test_path=test_path
)

# set model parameters
ngram_list = [1, 2]
target_feature = 'fog_q_class'
vector_type = 'tf-idf'
n_features = 250
augment_dict = {
    'token_length': True, 
    'patient_demographics': True, 
    'mds_updrs': True,
    'moca': True
}
seed = 13
k = 5
constant_dict = None # {'cache_size': 6_000}
hparam_optimizer.set_params(
    ngram_list=ngram_list, 
    target_feature=target_feature, 
    vector_type=vector_type, 
    n_features=n_features, 
    augment_dict=augment_dict, 
    seed=seed,
    k=k,
    constant_dict=constant_dict
)

# start the optimization loop
n_random = 50
n_guided = 100
opt_idx = '1'
hparam_optimizer.optimize(
    n_random=n_random, 
    n_guided=n_guided, 
    opt_idx=opt_idx
)