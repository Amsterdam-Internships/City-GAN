"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    opt_valid = TrainOptions().parse()
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)

    # Create a validation dataset
    opt_valid.phase = "val"
    opt_valid.num_threads = 0
    opt_valid.batch_size = opt.val_batch_size
    val_dataset = create_dataset(opt_valid)

    # get the number of images in the dataset.
    dataset_size = len(dataset)
    opt.dataset_size = dataset_size
    print(f'The number of training images = {dataset_size}')
    total_nr_epochs = opt.n_epochs + opt.n_epochs_decay + 1 - opt.epoch_count
    print(f'The number of epochs to run = {total_nr_epochs}')

    # set random seeds for reproducibility
    torch.manual_seed(opt.seed)

    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    # the total number of training iterations
    total_iters = 0

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # timer for entire epoch
        epoch_start_time = time.time()
        # timer for data loading per iteration
        iter_data_time = time.time()
        # the number of training iterations in current epoch
        epoch_iter = 0
        # reset the visualizer: make results are saved every epoch
        visualizer.reset()

        # inner loop within one epoch, iterating over batches
        for i, data in enumerate(dataset):
            # timer for computation per iteration
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # run everything on validation set every val_freq batches
            if total_iters % (opt.val_freq * opt.batch_size) == 0:
                if opt.verbose:
                    print("running validation set")
                model.run_validation(val_dataset)

            # this includes setting and preprocessing the data, and optimizing
            # the parameters
            model.run_batch(data, total_iters)

            # display images on visdom and save images to a HTML file
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, epoch_iter=epoch_iter)

            # print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                if opt.verbose:
                    # print discriminator scores on various batches
                    print(f"D preds: real {torch.mean(model.pred_real):.2f}, fake {torch.mean(model.pred_fake):.2f}, grfake: {torch.mean(model.pred_gr_fake):.2f}")
                    # print discriminator accuracies
                    print(f"accuracy: real: {model.acc_real}, fake: {model.acc_fake}, grfake: {model.acc_grfake}\n")

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            # TODO: check if saving the model can be done more effiently
            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # TODO: check if lr update here is correct (was before epoch)
        # update learning rate at the end of every epoch
        model.update_learning_rate()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    model.save_networks('latest')
    print("Finished training, model is saved")
