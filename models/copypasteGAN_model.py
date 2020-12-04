"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (batch, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(batch) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import torch.nn.functional as F
from models.base_model import BaseModel
import models.networks as networks

class CopyPasteGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        # comments for defaults:
        # images are resized to 64x64, hence load_size=70, crop_size=64
        # batch size of 80 per GPU is used (and 4 GPUs)
        # norm is instance by default (as in paper)
        # do not flip the images
        #

        # set default options for this model
        parser.set_defaults(dataset_mode='double', name="CopyGAN", load_size=70, crop_size= 64,batch_size=80, lr=1e-4, no_flip=True, lr_policy="step", direction=None, n_epochs=5, n_epochs_decay= 1,netG="copy", netD="copy", dataroot="datasets", save_epoch_freq=50, display_freq=1, print_freq=100)

        # define new arguments for this model
        if is_train:
            parser.add_argument('--lambda_aux', type=float, default=0.2, help='weight for the auxiliary mask loss')
            parser.add_argument('--confidence_weight', type=float, default=0.1, help='weight for the confidence loss for generator')
            parser.add_argument('--nr_obj_classes', type=int, default=1, help='Number of object classes in images, used for multiple masks')
            parser.add_argument('--D_head_start', type=int, default=1000, help='First train only discriminator for D_head_start iterations')
            # parser.add_argument('--multi_layered', action='store_true', default=3, help='Number of object classes in images, used for multiple masks')

        # nr_object_classes is used to output a multi-layered mask, each
        # channel representing a different object class


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

        self.multi_layered = opt.nr_obj_classes != 1
        self.D_head_start = opt.D_head_start

        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['loss_G_comp', 'loss_G_anti_sc', 'loss_G',
            'loss_D_real', 'loss_D_fake', "loss_D_gr_fake", "loss_AUX",
            "loss_D", "loss_G_conf"]
        if self.multi_layered:
            self.loss_names.append("loss_G_distinct")
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['src', 'tgt', 'g_mask', "g_mask_binary",
            'composite', "D_mask_fake", 'grounded_fake', "D_mask_grfake",
            'anti_sc_src', 'anti_sc', "D_mask_antisc", "D_mask_real"]

        # define generator, output_nc is set to nr of object classes
        self.netG = networks.define_G(opt.input_nc, opt.nr_obj_classes, ngf=opt.ngf, netG=opt.netG, norm=opt.norm, gpu_ids=self.gpu_ids, img_dim=opt.crop_size)
        # specify which models to save to disk
        self.model_names = ['G']

        if self.isTrain:
            # only define the Discriminator if in training phase
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, norm=opt.norm, gpu_ids=self.gpu_ids, img_dim=opt.crop_size)
            self.model_names.append("D")

            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode="vanilla").to(self.device)
            self.criterionMask = networks.MaskLoss().to(self.device)
            self.criterionConf = networks.ConfidenceLoss().to(self.device)
            if self.multi_layered:
                self.criterionDist = networks.DistinctMaskLoss(opt.nr_obj_classes).to(self.device)

            # define optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """

        self.src = input['src'].to(self.device)  # get image data
        self.tgt = input['tgt'].to(self.device)

        # create a grounded fake, the function samples a random polygon mask
        self.grounded_fake, self.mask_gf = networks.composite_image(self.src, self.tgt, device=self.device)



    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        # generate output image given the input batch
        self.g_mask_layered = self.netG(self.src)
        self.g_mask = torch.max(self.g_mask_layered, dim=1, keepdim=True)[0]

        # binary mask for visualization
        self.g_mask_binary = networks.mask_to_binary(self.g_mask)

        # create the composite mask from src and tgt images, and predicted mask
        self.composite, _ = networks.composite_image(self.src, self.tgt, self.g_mask, device=self.device)

        # TODO: is this sound to create anti shortcut?
        # apply the masks on different source images, should be labeled false
        # we reverse the src images over the batch dimension
        self.anti_sc_src = torch.flip(self.src, [0, 1])
        self.anti_sc, _ = networks.composite_image(self.anti_sc_src, self.tgt, self.g_mask)


    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration. Discriminator predictions have been computed
        in the backward_D"""

        # stimulate the generator to fool discriminator
        self.loss_G_comp = self.criterionGAN(self.pred_fake, True)
        self.loss_G_anti_sc = self.criterionGAN(self.pred_anti_sc, False)
        self.loss_G_conf = self.opt.confidence_weight * self.criterionConf(
            self.g_mask)


        # add up components and compute gradients
        self.loss_G = self.loss_G_comp + self.loss_G_anti_sc + \
            self.loss_G_conf

        if self.multi_layered:
            self.loss_G_distinct = self.criterionDist(self.g_mask_layered)
            self.loss_G = self.loss_G + self.loss_G_distinct

        self.loss_G.backward()


    def backward_D(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.composite has been computed during function <forward>
        torch.autograd.set_detect_anomaly(True)
        # get predictions from discriminators for all images
        self.pred_real, self.D_mask_real = self.netD(self.tgt) # can also be source
        self.pred_fake, self.D_mask_fake = self.netD(self.composite)
        self.pred_gr_fake, self.D_mask_grfake = self.netD(self.grounded_fake)
        self.pred_anti_sc, self.D_mask_antisc = self.netD(self.anti_sc)

        # compute the GAN losses
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_gr_fake = self.criterionGAN(self.pred_gr_fake, False)

        # compute auxiliary loss, directly use lambda for plotting purposes
        self.loss_AUX = self.opt.lambda_aux * self.criterionMask(
            self.D_mask_real, self.D_mask_fake, self.D_mask_antisc,
            self.D_mask_grfake, self.g_mask, self.mask_gf)

        # sum the losses
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gr_fake + self.loss_AUX

        # Calculate gradients of discriminator
        self.loss_D.backward(retain_graph=True)


    def optimize_parameters(self, total_iters):
        """Update network weights; it is called in every training iteration.
        only perform  optimizer steps after all backward operations, torch1.5
        gives an error, see https://github.com/pytorch/pytorch/issues/39141

        Arguments:
            - total_iters: training progress in steps, used to give D a
            headstart

        """
        train_G = total_iters > self.D_head_start

        # perform forward step
        self.forward()

        # reset previous gradients and compute new gradients for D
        self.optimizer_D.zero_grad()
        self.backward_D()

        # only train G after headstart for D
        if self.train_G:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        # update D
        self.optimizer_D.step()







