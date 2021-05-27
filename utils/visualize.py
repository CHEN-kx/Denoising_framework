import os
import numpy as np
import skimage as io
from tensorboardX import SummaryWriter

def create_writer(name):
    # logs, image, diff, spec
    os.makedirs('results/%s/logs' % name, exist_ok=True)
    os.makedirs('results/%s/image' % name, exist_ok=True)
    writer = SummaryWriter("results/%s/logs" % name)
    return writer

def add_scalar_summary(writer, step, name, data, stage="train"):
    writer.add_scalars("scalars/%s" % stage, {name: data}, step)

def save_image(name, filename, data, normalize=True, permute=True):
    data = np.clip(data, 0, 1) ** 0.45454545 if normalize else data
    data = data.permute(1, 2, 0) if permute else data
    img = data.data.cpu().numpy()
    img = np.uint8(img * 255)
    io.imsave(os.path.join("results/%s/image" % name, "%s.png" % filename), img)