
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import copy

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)

    opt_valid = copy.copy(opt)
    opt_valid.phase = "val"
    # opt_valid.num_threads = 0
    opt_valid.batch_size = opt.val_batch_size
    val_dataset = create_dataset(opt_valid)


    # # get the number of images in the dataset.
    dataset_size = len(dataset)
    opt.dataset_size = dataset_size

    assert opt.model in {"copy", "move", "classifier"}

    print(f"Starting training of {opt.model}-model")
    print(f'The number of training images = {dataset_size}')
    print(f'The number of validation images = {len(val_dataset)}')
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
    overall_batch = 0
    D_fakes = []


    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # timer for entire epoch
        epoch_start_time = time.time()
        # timer for data loading per iteration
        iter_data_time = time.time()
        # the number of training iterations in current epoch
        epoch_iter = 0
        epoch_batch = 0
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
            epoch_batch += 1
            overall_batch += 1

            # run everything on validation set every val_freq batches
            # also run the untrained model (batch = 0), for baseline
            if (overall_batch -1) % opt.val_freq == 0 and opt.model != "classifier":
                model.eval()
                val_start_time = time.time()
                model.run_validation(val_dataset)
                if opt.verbose:
                    duration = time.time() - val_start_time
                    print(f"ran validation set (B:{overall_batch}) in \
                        {duration:.1f} s.")
                model.train()


            # inference can be called here, make another script TODO
            # original, moved = model.baseline(data, type_="random")
            model.train()
            model.run_batch(data, overall_batch)

            # display images on visdom and save images to a HTML file
            if overall_batch % opt.display_freq == 0:
                if opt.model != "classifier":
                    D_fake = model.pred_fake[0]
                    D_fake = D_fake.detach().squeeze().cpu().numpy().round(2)
                    D_fakes.append(D_fake)
                else:
                    D_fakes = None
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, overall_batch=overall_batch, D_fakes=D_fakes)
                if opt.model=='move':
                    print(overall_batch, model.theta_complete[0])

            # print training losses and save logging information to the disk
            # Why is this epoch_batch? now set to overall_batch
            if epoch_batch % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_batch, losses, t_comp, t_data)

                # get some insight in the gradient scaling process for copyGAN
                if opt.use_amp:
                    print(model.scaler.state_dict())

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)


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
    if opt.model != "classifier":
        print(f"Batches trained - G: {model.count_G}, D: {model.count_D} ")
