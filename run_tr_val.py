import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "allocator=pluggable,min_split_size_mb=16,max_split_size_mb=256"
from train_test import train_val_multi
from args2 import config

print(config)
train_val_multi(config)
