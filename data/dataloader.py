import os
import torch
import torch.utils.data

def to_torch_tensors(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                data[k] = torch.from_numpy(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if not isinstance(v, torch.Tensor):
                data[i] = to_torch_tensors(v)
    return data

def load_data(cfg, is_train):
    if is_train:
        dataset = renderDataset(cfg.train_data_path, 'train')
    else:
        dataset = renderDataset(cfg.valid_data_path, 'valid')

    batch_size = cfg.batch_size if is_train else 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=cfg.numworks)
    return loader
    

data_dict = {'diff':['X_diff, Reference','albedo'], 'spec':['X_spec, Reference'], 'eval':['X','Y']}

class renderDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, stage, type):
        self.data_path, self.stage = data_path, stage
        self.patches = sorted(os.listdir(self.data_path))
        self.patch_num = len(self.patches) 
        self.data_type = data_dict[type][:2] if self.stage is 'train' else data_dict[type]

    def __len__(self):
        return self.patch_num       

    def __getitem__(self, idx):
        patch_path = os.path.join(self.data_path, self.patches[idx])
        patch_name = self.patches[idx].split('.')[0]
        patch = to_torch_tensors(torch.load(patch_path))
        data = [patch[i] for i in self.data_type]
        return data if self.stage is 'train' else (data, patch_name)
