"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import numpy as np
from .base_model import BaseModel
from util import util
from . import networks

from torchvision.transforms.functional import affine
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import random


class MoveModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='room', preprocess="resize", load_size=64, crop_size=64, no_flip=True, netD='basic', init_type="normal", name="MoveModel", lr_policy="step", gan_mode="vanilla")  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--theta_dim', type=int, default=2, help=
                "specify how many params to use for the affine tranformation. Either 6 (full theta) or 2 (translation only)")


        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.loss_names = ["loss_D_real", "loss_D_fake",  "loss_D", "loss_Conv", "acc_real", "acc_fake", "scaled_theta_X", "scaled_theta_Y"]

        for loss in self.loss_names:
            setattr(self, loss, 0)

        # define variables for plotting and saving
        self.visual_names = ["tgt", "src", "mask_binary", "obj", "composite"]

        self.scaler = GradScaler()

        if opt.tracemalloc:
            import tracemalloc

        # define the convnet that predicts theta
        # perhaps we should treat the object and target separately first
        self.netConv = networks.define_D(opt.input_nc * 2, opt.ngf, netD="move", n_layers_D=opt.n_layers_conv, gpu_ids=self.gpu_ids, norm=opt.norm, init_type=opt.init_type, theta_dim=opt.theta_dim)

        # self.netConv = networks.MoveConvNET(opt.input_nc*2, opt.ngf, n_layers=opt.n_layers_conv, norm=opt.norm, theta_dim=opt.theta_dim)

        self.model_names = ["Conv"]

        if self.isTrain:
            # define Discriminator
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, gpu_ids=self.gpu_ids)
            self.model_names.append("D")

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, target_real_label=opt.real_target).to(self.device)

            # define optimizers
            self.optimizer_Conv = torch.optim.Adam(
                self.netConv.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))
            self.optimizers = [self.optimizer_Conv, self.optimizer_D]


        # The program will automatically call <model.setup> to define schedulers, load networks, and print networks



    def center_object(self, mask):
        """
        This function performs the following steps:
        - A valid mask from the mask_dict is selected
        - Mask is used to extract the object from the source image
        - The object and corresponding mask are centered
        """

        # get the surface per mask in the batch
        surface = self.mask_binary.sum([2, 3]).view(-1, 1, 1, 1)
        # center and return the object
        # inspired on https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
        mask_pdist = self.mask_binary / surface

        [self.B, _, self.w, self.h] = list(mask_pdist.shape)
        # assert B == self.opt.batch_size, f"incorrect batch dim: {B}"

        x_center, y_center = self.w//2, self.h//2


        # marginal distributions
        dx = torch.sum(mask_pdist, 2).view(-1, self.w)
        dy = torch.sum(mask_pdist, 3).view(-1, self.h)


        # expected values
        cx = torch.sum(dx * torch.arange(self.w).to(self.device), 1)
        cy = torch.sum(dy * torch.arange(self.h).to(self.device), 1)

        # print("cx, cy", cx, cy)

        # compute necessary translations
        x_t = x_center - cx
        y_t = y_center - cy

        # print("x_t, y_t", x_t, y_t)

        # extract object from src
        obj = (self.mask_binary) * self.src

        # translate the object and mask
        obj_centered = torch.stack([affine(o, shear=0, translate=[x, y], scale=1, angle=0) for o, x, y in zip(obj, x_t, y_t)], 0)

        obj_mask = torch.stack([affine(m, shear=0, translate=[x, y], scale=1, angle=0) for m, x, y in zip(self.mask_binary, x_t, y_t)], 0)

        return obj_centered, obj_mask


    def set_input(self, input):
        """
        What do we need:

            - centered object + object mask (from source)
            - some real image (e.g. source)
            - target image

        """

        self.src = input['src'].to(self.device)
        self.tgt = input['tgt'].to(self.device)
        self.mask = input['mask'].to(self.device)

        self.mask_binary = (self.mask > 0).int().to(self.device)

        # find a suitable object to move from src to target
        self.obj, self.obj_mask = self.center_object(self.mask)



    def forward(self, valid=False):
        """
        what needs to be done:
            - target image and centered object are fed to convnet (initialized in the init)
            - theta parameters are the output
            - affine transformation on the object and the object mask
            - the transformed object and object masks are composited --> output img
        """
        # concatenate the target and object on channel dimension
        tgt_obj_concat = torch.cat([self.tgt, self.obj], 1)

        # compute theta using the convolutional network
        self.theta = self.netConv(tgt_obj_concat).squeeze()

        # print("theta:", self.theta)
        # make sure theta is scaled: preventing object from moving outside img

        self.scaled_theta = (self.theta * torch.tensor([self.w//2, self.h//2]).to(self.device)).int().view(-1, self.opt.theta_dim)

        # for plotting and printing purposes
        self.scaled_theta_X = self.scaled_theta[0][0]
        self.scaled_theta_Y = self.scaled_theta[0][1]

        # print("scaled_theta:", self.scaled_theta)

        # use theta to transform the object and the mask
        # is affine the correct function? perhaps we should use affine_grid

        self.transf_obj = torch.stack([affine(obj, translate=list(theta), angle=0, scale=1, shear=0) for obj, theta in zip(self.obj, self.scaled_theta)], 0)
        self.transf_obj_mask = torch.stack([affine(mask, translate=list(theta), angle=0, scale=1, shear=0) for mask, theta in zip(self.obj_mask, self.scaled_theta)], 0)


        # composite the moved object with the background from the target
        self.composite, _ = networks.composite_image(self.transf_obj, self.tgt, self.transf_obj_mask)

        # get the prediction on the fake image
        self.pred_fake = self.netD(self.composite)

        if valid:
            self.pred_real = self.netD(self.src)
            self.compute_accs()


    def compute_accs(self):
        [B, _, d, _] = self.pred_real.shape

        total = B * d ** 2

        self.acc_real = len(self.pred_real[self.pred_real>0.5]) / total
        self.acc_fake = len(self.pred_fake[self.pred_fake<0.5]) / total


    def backward_D(self):
        """
        todo here
            - compute the losses (GAN loss real fake and possibly more)
            - backward over the loss for D
        """

        # get the prediction on the real image
        self.pred_real = self.netD(self.src)


        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2

        # self.scaler.scale(self.loss_D).backward()
        self.loss_D.backward()



    def backward_Conv(self):
        """
        todo here
            - compute the losses (GAN loss real fake and possibly more)
            - backward over the loss to update the ConvNet

        """

        self.loss_Conv = self.criterionGAN(self.pred_fake, True)

        # self.scaler.scale(self.loss_Conv).backward()
        self.loss_Conv.backward()


    def optimize_parameters(self, data, overall_batch):
        """Update network weights; it will be called in every training iteration.
        """

        self.set_input(data)

        # run the forward pass
        self.forward()

        # train discriminator
        if overall_batch % 3 == 0:
            # print("Training D")
            self.optimizer_D.zero_grad()
            self.backward_D()
            # self.scaler.step(self.optimizer_D)
            # self.scaler.update()
            self.optimizer_D.step()

        # train convnet predicting theta
        else:
            # print("Training Convnet")
            self.optimizer_Conv.zero_grad()
            self.backward_Conv()
            # self.scaler.step(self.optimizer_Conv)
            # self.scaler.update()
            self.optimizer_Conv.step()



    def baseline(self, data, type_='random'):

        assert type_ in {"random", "scanline"}, f"Type {type_} not recognized, choose from \"random\" or \"scanline\""

        assert self.opt.batch_size == 1,"for baselines, batch size should be 1"

        self.set_input(data)

        img_width, img_height = self.src.shape[2:4]

        obj_width = int(torch.max(torch.sum(self.obj_mask>0, axis=2)))
        obj_height = int(torch.max(torch.sum(self.obj_mask>0, axis=3)))

        obj_size_approx = obj_width * obj_height

        # divide the image into segments
        # background includes the object to be moved
        background = (1-self.mask_binary) * self.src
        obj = self.mask_binary * self.src


        # x translation is always used
        # x_translation = np.random.randint(obj_width, img_width - obj_width)
        x_translation = (1 if np.random.random() < 0.5 else -1) * obj_width
        # x_translation = np.random.normal(obj_width, 2*obj_width)

        if type_=="random":
            y_translation = (1 if np.random.random() < 0.5 else -1) * obj_height

        elif type_=="scanline":
            # we want to obtain x_min and x_max of the object, and the width of the object (x_max-x_min). Then we move the object to the right with at least width, and maximally (img_width - width), and modulo the transformation with img_width

            if obj_width >= img_width:
                print("object is too large, to be implemented (returns None)")
                return None, None

            y_translation = 0

        moved_obj = affine(obj, 0, [x_translation, y_translation], 1, 0)
        new_background = 1 - (moved_obj != 0).int()
        self.moved = new_background  * self.src + moved_obj

        print(x_translation, y_translation)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(util.tensor2im(self.src), origin="upper")
        ax1.set_title("original")
        ax2.imshow(util.tensor2im(obj), origin="upper")
        ax2.set_title("object")
        ax3.imshow(util.tensor2im(moved_obj), origin="upper")
        ax3.set_title(f"moved_obj ({x_translation}, {y_translation})")
        ax4.imshow(util.tensor2im(self.moved), origin="upper")
        ax4.set_title("result")
        plt.tight_layout()
        plt.show()


        return self.tgt, self.moved


    def run_validation(self, val_data):
        """
        Run the complete validation set, and set booleans describing the model
        performance. These are used to determine the training schedule.

        Arguments:
            - val_data: batch of validation data
        """

        if self.opt.tracemalloc:
            tracemalloc.start()

        # init average lists
        acc_real = []
        acc_fake = []

        # compute accuracy on the validation data
        with torch.no_grad():
            for i, data in enumerate(val_data):
                # preprocess data and perform forward pass
                self.set_input(data)
                with autocast():
                    self.forward(valid=True)

                # save accuracies
                acc_fake.append(self.acc_fake)
                acc_real.append(self.acc_real)

        # set accuracies to mean for plotting purposes
        self.acc_fake = np.mean(acc_fake)
        self.acc_real = np.mean(acc_real)


        # print validation scores
        if self.opt.verbose:
            print(
                f"validation accuracies:\n\
                real: {self.acc_real:.2f}\n\
                fake: {self.acc_fake:.2f}\n"
            )

        if self.opt.tracemalloc:
            snapshot = tracemalloc.take_snapshot()
            print("Tracemalloc: validation memory")
            util.print_snapshot(snapshot)
