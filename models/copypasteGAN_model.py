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
        # images are resized to 64x64, hence load_size=64
        # batch size of 80 per GPU is used (and 4 GPUs)

        parser.set_defaults(dataset_mode='aligned', output_nc=1, load_size=70, crop_size= 64,batch_size=80, lr=1e-4, lr_policy="step", n_epochs=1, netG="copy", netD="copy", dataroot="datasets")  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_aux', type=float, default=0.2, help='weight for the auxiliary mask loss')  # You can define new arguments for this model.

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

        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['loss_G', "loss_D", "loss_AUX"]
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # only define the Discriminator if in training phase
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[])
            self.model_names.append("D")

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            self.criterionGAN = networks.GANLoss(gan_mode="vanilla").to(self.device)
            self.criterionShortCut = networks.MaskLoss().to(self.device)
            self.criterionMask = networks.MaskLoss().to(self.device)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        print("Data_A", self.data_A.shape)
        self.output = self.netG(self.data_A)  # generate output image given the input data_A


    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        # second argfument of critergion was set to dataB
        self.loss_G_fake = self.criterionGAN(self.output, False)
        self.loss_G_antishortcut = 0

        # add up components and compute gradients
        self.loss_G = self.loss_G_fake + self.loss_G_antishortcut
        self.loss_G.backward()

    def backward_D(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        # self.loss_D_real = self.criterionGAN(self.output, self.data_B)

        self.loss_D_real = self.criterionGAN(self.output, True)
        self.loss_D_fake = self.criterionGAN(self.output, False)
        self.loss_D_groundedfake = self.criterionGAN(self.grounded_fake, False)
        self.loss_AUX = self.criterionMask()

        self.loss_D = self.loss_D_real + self.loss_D_fake + loss_D_groundedfake + self.opt.lambda_aux * self.loss_AUX

        # Calculate gradients of discriminator
        self.loss_D.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""

        # perform forward step
        self.forward()

        # clear existing gradients
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # compute gradients
        self.backward_G()
        self.backward_D()

        # update the networks
        self.optimizer_G.step()
        self.optimizer_D.step()






