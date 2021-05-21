"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import linecache
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def get_visuals_copy(opt, isTrain, aux):
    # specify the images that are saved and displayed
    # (via base_model.get_current_visuals)
    if not isTrain:
        visual_names =[
                "src",
                "tgt",
                "composite",
                "g_mask",
                "g_mask_binary",
                "gt",
                "gt_og",
                "used_comb_gt",]
                # "eroded_mask",
                # "composite_eroded",
                # "labelled_mask"]
    else:
        if aux:
            visual_names = [
                "src",
                "tgt",
                "g_mask",
                "g_mask_binary",
                "composite",
                "D_mask_fake",
                "irrel",
                "anti_sc",
                "D_mask_antisc",
                "D_mask_real",
            ]
            if not opt.no_grfakes:
                visual_names.extend(
                    ["grounded_fake", "D_mask_grfake", "mask_gf"]
                )
        else:
            visual_names = [
                "src",
                "tgt",
                "g_mask",
                "g_mask_binary",
                "composite",
                "irrel",
                "anti_sc",
            ]
            if not opt.no_grfakes:
                visual_names.extend(["grounded_fake", "mask_gf"])



    return visual_names


def print_gradients(net):
    print(f"{net.__class__.__name__}")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f"name: {name}, norm gradient: {param.grad.norm():.5f}")

def plot_confusion_matrix(conf_matrix, labels, save_path):
    plt.figure(figsize=(12, 12))
    cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    cmd.plot()
    plt.savefig(save_path)


def mask_to_binary(mask):
    """
    Convert a mask in [0, 1] to binary ( in {0, 1})
    """

    assert (mask.min().item() >= 0) and (mask.max().item() <= 1)
    bin_mask = F.relu(torch.sign(mask - 0.5))

    return bin_mask


def split_mask(mask):
    """
    extract the number of different values for
    the b_channel (3rd channel)
    the output is a 4D tensor, with each mask over the batch dimension
    """

    assert mask.shape[0] == 3
    dim = mask.shape[-1]

    # extract the blue channel
    b_channel = mask[2, :, : ]
    # find the unique values in the tensor, these correspond to the masks
    values = torch.unique(b_channel)
    # initialize the 4D output tensor
    out = torch.zeros(values.size()[0], 1, dim, dim)

    for i, v in enumerate(values):
        # skip background
        if v == 0:
            continue
        # extract the object mask from blue channel
        binary_mask = torch.where(b_channel == v, 1, 0)
        # set in output batch
        out[i, 0, :, :] = binary_mask

    return out


def compute_accs(self):
    """
    Computes the accuracies of the discriminator based on patch predictions
    NB: self is here the model class, to access all predictions. Not used atm,
    should be moved to inside the model class to be used.
    """

    # assign the fakest patch in case of patch discriminator, else use the scaler prediction
    patch = self.pred_real_patch.dim() > 2
    fakest_patch_fake = (
        torch.amin(self.pred_fake, dim=(2, 3)) if patch else self.pred_fake
    )
    fakest_patch_real = (
        torch.amin(self.pred_real, dim=(2, 3)) if patch else self.pred_real
    )

    # predictions above 0.5 are classified as "real"
    B = self.opt.val_batch_size
    self.acc_real = len(fakest_patch_real[fakest_patch_real > 0.5]) / B
    self.acc_fake = len(fakest_patch_fake[fakest_patch_fake < 0.5]) / B

    if self.train_on_gf:
        fakest_patch_grfake = (
            torch.amin(self.pred_grfake, dim=(2, 3))
            if patch
            else self.pred_grfake
        )
        self.acc_grfake = (
            len(fakest_patch_grfake[fakest_patch_grfake < 0.5]) / B
        )



def print_snapshot(snapshot):
    """
    can be used to print snapshot from tracemalloc
    snapshot can be taken using tracemalloc.take_snapshot()
    """
    top_stats = snapshot.statistics('lineno')

    for index, stat in enumerate(top_stats[:3], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
