# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-12-04 09:38
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-12-16 16:33


"""
This script takes a json file with losses with the following structure:

{
  "(epoch) 1": {
    "(iter) 1000": {
      "loss_G_comp": 0.616,
      ...
      "loss_G_conf": 0.023
    },
    "(iter) 2000": {
        ...
    }
    ...
  }
  "(epoch) 2": {
    ...
  }
  ...
}

And saves a plot to destination filename

"""


import argparse
import matplotlib.pyplot as plt
import json
import os
from glob import glob
import numpy as np


def plot_json(opt):
    assert os.path.exists(opt.filename), "JSON file could not be found"

    with open(opt.filename) as f:
        data = json.load(f)

    n_epochs = len(data.keys())
    iters_per_epoch = max([int(i) for i in data["1"].keys()])
    loss_names = list(data['1'][str(iters_per_epoch)].keys())
    step_size = min([int(i) for i in data["1"].keys()])
    max_iter = n_epochs*iters_per_epoch + step_size
    all_iters = np.arange(step_size, max_iter, step_size)


    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(7, 7))

    for loss_name in loss_names:
        losses = [data[epoch][iter_][loss_name] for epoch in data.keys() for iter_ in data['1']]
        losses = running_mean(losses, opt.n)
        plot_iters = all_iters[opt.n//2:-(opt.n//2)]

        if "acc" in loss_name:
            label = loss_name[4:]
            ax1.plot(plot_iters, losses, label=label)
        else:
            label = loss_name[5:]
            ax2.plot(plot_iters, losses, label=label)


    fig.suptitle(f'Accuracy and loss plot run {opt.run}')
    ax2.set_title("Losses")
    ax1.set_title("Discriminator Accuracy")
    ax2.set(xlabel=f"Iteration ({iters_per_epoch} per epoch)")
    ax1.set(ylabel="Accuracy")
    ax2.set(ylabel="Loss")
    ax1.legend()
    ax2.legend()
    ax1.label_outer()
    ax2.label_outer()

    plt.tight_layout()

    plt.savefig(opt.dest)

    print(f"figure saved to {opt.dest}")









def running_mean(vals, n=5):
    assert n < len(vals)
    out = np.convolve(vals, np.ones(n)/n, mode='valid')
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="",
        help='path to json file to be plotted')
    parser.add_argument('--run', type=int, default=0,
        help='run to plot json file from, default 0 will plot latest')
    parser.add_argument('--dest', type=str, default="",
        help='leave empty to save plot at filename path')
    parser.add_argument('--n', type=int, default=5,
        help='Interval to take running mean over for plotting')

    opt, unparsed = parser.parse_known_args()

    if not opt.filename:
        if opt.run == 0:
            # latest run is extracted
            opt.run = sorted(glob("checkpoints/run*"))[-1][-1]
        opt.filename = f"checkpoints/run{opt.run}/checkpoints/CopyGAN/loss_log.json"


    if not opt.dest:
        opt.dest = f"checkpoints/run{opt.run}/checkpoints/CopyGAN/loss_plot.pdf"


    plot_json(opt)






