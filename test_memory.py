import torch
import os
import time
train_path = os.path.join('/nfs/ckx/denoising/kpcn/', "train_data", 'diff')
valid_path = os.path.join('/nfs/ckx/denoising/kpcn/', "valid_data", 'diff')

t = time.time()
tarin_patch = [torch.load(os.path.join(train_path, i)) for i in os.listdir(train_path)]
print("train!")
print('The time of train data : %f' %(time.time()-t))
t = time.time()
valid_patch = [torch.load(os.path.join(valid_path, i)) for i in os.listdir(valid_path)]
print('test!')
print('The time of test data : %f' %(time.time()-t))