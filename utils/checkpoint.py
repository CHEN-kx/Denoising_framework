import os
import shutil
import torch

def save_checkpoint(model_dir, state, is_best):
    """save checkpoint"""
    # write model-last
    epoch = state['epoch']
    path = os.path.join(model_dir, 'model-last.pth')
    torch.save(state, path)

    # write checkpoint
    checkpoint_file = os.path.join(model_dir, 'checkpoint')
    checkpoint = open(checkpoint_file, 'w+')
    checkpoint.write('Epoch%-4d: model_checkpoint_path:%s\n' %(epoch,path))
    checkpoint.close()

    # write model-best
    if is_best:
        shutil.copyfile(path, os.path.join(model_dir, 'model-best.pth'))


def load_state(model_dir, model, optimizer=None):
    """load state"""
    if not os.path.exists(model_dir + '/checkpoint'):
        print("=> no checkpoint found at '{}', train from scratch".format(model_dir))
        return 0, 0
    else:
        ckpt = open(model_dir + '/checkpoint')
        model_path = ckpt.readlines()[0].split(':')[1].strip('\n')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('missing keys from checkpoint {}: {}'.format(model_dir, k))

        print("=> loaded model from checkpoint '{}'".format(model_dir))

        if optimizer != None:
            best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})"
                  .format(model_dir, start_epoch))
            return best_prec1, start_epoch


def load_state_ckpt(model_path, model):
    """load state checkpoint"""

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('missing keys from checkpoint {}: {}'.format(model_path, k))

    print("=> loaded model from checkpoint '{}'".format(model_path))


def load_state_epoch(model_dir, model, epoch):
    """load state epoch"""

    model_path = model_dir + '/model.pth-' + str(epoch)
    load_state_ckpt(model_path, model)