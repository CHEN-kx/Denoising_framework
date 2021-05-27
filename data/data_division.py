import os
import torch

def prepare_diff_patch(patch):
    diff_p={}
    diff_p['X_diff'] = patch['X_diff']#10
    diff_p['Y_diff'] = patch['Reference'][:,:,:3]#3
    diff_p['albedo'] = patch['origAlbedo']#3
    return diff_p

def prepare_spec_patch(patch):
    spec_p={}
    spec_p['X_spec'] = patch['X_spec']#10
    spec_p['Y_spec'] = patch['Reference'][:,:,3:]#3
    return spec_p

def prepare_eval_patch(patch):
    p={}
    p['X'] = patch['finalInput']#3
    p['Y'] = patch['finalGt']#3
    return p

if __name__ == '__main__':
    for x in ['train_data','valid_data']:
        data_path = "/nfs/ckx/Kan/128x128p30/"+ x
        outputpath = "/nfs/ckx/denoising/kpcn/"+ x
        patchlist = os.listdir(data_path)
        for i in patchlist:
            patch = torch.load(os.path.join(data_path,i))
            torch.save(prepare_diff_patch(patch), outputpath+"/diff/"+i)
            torch.save(prepare_spec_patch(patch), outputpath+"/spec/"+i)
            torch.save(prepare_eval_patch(patch), outputpath+"/eval/"+i)
            print(i)
