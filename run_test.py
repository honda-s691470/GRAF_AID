import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "allocator=pluggable,min_split_size_mb=16,max_split_size_mb=256"
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import argparse
#import yaml
#from utils import * 
from train_test import test #prepare_trte_data_multi, gen_trte_adj_mat, gen_trte_adj_mat_test, train_val_multi, test
#cuda = True if torch.cuda.is_available() else False
from args2 import config

if len(config["adj_parameter"]) == 1:
    config["adj_parameter"] = config["adj_parameter"][0]

print(config)
adj_param = test(config)
