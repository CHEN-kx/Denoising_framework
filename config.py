import os


''' experiment config '''
gpu = '3'
mode = 'srvc'
ckpt = ""
name = "test_code" 
save = os.path.join("results", name)


''' data config '''
type = "diff" 
data_path = '/nfs/ckx/denoising/kpcn/'
train_data_path = os.path.join(data_path, "train_data", type)
valid_data_path = os.path.join(data_path, "valid_data", type)

patch_size = 128 
batch_size = 12
numworks = 12


''' trainning config '''
loss = "l1"
lr = 1e-4
eps, inf = 0.00316, 1e30
permu = [0, 3, 1, 2]  # NHWC -> NCHW
epoch_num = 300
save_epoch = 1


''' model config '''
norm_type = 'none' #'batch' 'none' 'instance'
input_nc_x = 3
input_nc_ft = 7
input_nc = [input_nc_x, input_nc_ft]
hidden_nc = 64
n_blocks = 16


''' kpn model config '''
k = 0


''' srvc model config '''
spacesize = 5
head_nc = 8
adaconv_nc = 64
out_nc = 32