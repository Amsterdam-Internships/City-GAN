""" CopyPasteGAN model template

You can specify '--model copypasteGAN' to use this model.

"""


import torch
import torch.nn.functional as F
from models.base_model import BaseModel
import models.networks as networks

class CopyPasteGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for
        existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use
            this flag to add training-specific or test-specific options.

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
        parser.set_defaults(dataset_mode='double', name="CopyGAN",
            load_size=70, crop_size= 64, batch_size=50, lr=1e-4, no_flip=True,
            lr_policy="step", direction=None, n_epochs=1, n_epochs_decay=3,
            netG="copy", netD="copy", dataroot="datasets", save_epoch_freq=50,
            display_freq=1, print_freq=100)

        # define new arguments for this model
        if is_train:
            parser.add_argument('--lambda_aux', type=float, default=0.2,
                help='weight for the auxiliary mask loss')
            parser.add_argument('--confidence_weight', type=float, default=0.0,
                help='weight for the confidence loss for generator')
            parser.add_argument('--nr_obj_classes', type=int, default=1, help=
                'Number of object classes in images, used for multiple masks')
            parser.add_argument('--D_headstart', type=int, default=80000, help=
                'First train only discriminator for D_headstart iterations \
                (images, independent on batchsize')
            parser.add_argument('--beta2', type=int, default=0.999,
                help='beta2 parameter for the adam optimizer')
            parser.add_argument('--sigma_blur', type=float, default=1.0,
                help='Sigma used in Gaussian filter used for blurring \
                discriminator input')
            parser.add_argument('--real_target', type=float, default=1.0,
                help='Target label for the discriminator, can be set <1 to \
                prevent overfitting')
            parser.add_argument('--seed', type=int, default=42, help=
                'Provide an integer for setting the random seed')
            parser.add_argument('--border_zeroing', action='store_false', help=
                'default: clamp borders of generated mask to 0 (store_false)')
            parser.add_argument('--D_threshold', default=0.5, help=
                "when the accuracy of the discriminator is lower than this \
                threshold, only train D")

        # nr_object_classes is used to output a multi-layered mask, each
        # channel representing a different object class


        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        """
        BaseModel.__init__(self, opt)

        self.multi_layered = opt.nr_obj_classes != 1
        self.D_headstart = opt.D_headstart

        # specify random seed
        torch.manual_seed(opt.seed)

        # specify the training losses you want to print out.
        # base_model.get_current_losses is used for plotting and saving these
        self.loss_names = ['loss_G_comp', 'loss_G_anti_sc', 'loss_G',
            'loss_D_real', 'loss_D_fake', "loss_D_gr_fake", "loss_AUX",
            "loss_D", "acc_real", "acc_fake", "acc_grfake"]
        # add confidence loss if specified
        if opt.confidence_weight > 0:
            self.loss_names.append("loss_G_conf")

        # for visualization purposes, set G losses to zero in case of headstart
        if self.D_headstart > 0:
            self.loss_G_comp = self.loss_G_conf = self.loss_G_anti_sc = self.loss_G = 0
            if opt.confidence_weight > 0:
                self.loss_G_conf = 0

        self.train_on_gf = True
        self.acc_grfake = 0.0
        self.acc_fake = 0.0

        if self.multi_layered:
            self.loss_names.append("loss_G_distinct")

        # specify the images you want to save and display (via
        # base_model.get_current_visuals)
        self.visual_names = ['src', 'tgt', 'g_mask', "g_mask_binary",
            'composite', "D_mask_fake", 'grounded_fake', "mask_gf",
            "D_mask_grfake", 'anti_sc_src', 'anti_sc', "D_mask_antisc",
             "D_mask_real"]

        # define generator, output_nc is set to nr of object classes
        self.netG = networks.define_G(opt.input_nc, opt.nr_obj_classes,
            ngf=opt.ngf, netG=opt.netG, norm=opt.norm,
            border_zeroing=opt.border_zeroing, gpu_ids=self.gpu_ids,
            img_dim=opt.crop_size)

        # specify which models to save to disk
        self.model_names = ['G']

        if self.isTrain:
            # only define the discriminator if in training phase
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                norm=opt.norm, gpu_ids=self.gpu_ids, img_dim=opt.crop_size,
                sigma_blur=opt.sigma_blur)
            self.model_names.append("D")

            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode="vanilla",
                target_real_label=opt.real_target).to(self.device)
            self.criterionMask = networks.MaskLoss().to(self.device)
            self.criterionConf = networks.ConfidenceLoss().to(self.device)
            if self.multi_layered:
                self.criterionDist = networks.DistinctMaskLoss(
                    opt.nr_obj_classes).to(self.device)

            # define optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers = [self.optimizer_G, self.optimizer_D]



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata
            information.
        """

        # put image data on device
        self.src = input['src'].to(self.device)
        self.tgt = input['tgt'].to(self.device)

        # create a grounded fake, the function samples a random polygon mask
        if self.train_on_gf:
            self.grounded_fake, self.mask_gf = networks.composite_image(
                self.src, self.tgt, device=self.device)



    def forward(self):
        """Run forward pass. This will be called by both functions <
        optimize_parameters> and <test>."""

        # generate output image given the input batch
        self.g_mask_layered = self.netG(self.src)
        self.g_mask = torch.max(self.g_mask_layered, dim=1, keepdim=True)[0]

        # binary mask for visualization
        self.g_mask_binary = networks.mask_to_binary(self.g_mask)

        # create the composite mask from src and tgt images, and predicted mask
        self.composite, _ = networks.composite_image(self.src, self.tgt,
            self.g_mask, device=self.device)

        # apply the masks on different source images, should be labeled false
        # we reverse the src images over the batch dimension
        self.anti_sc_src = torch.flip(self.src, [0, 1])
        self.anti_sc, _ = networks.composite_image(self.anti_sc_src, self.tgt,
            self.g_mask)

        # get predictions from discriminators for all images (use tgt/src)
        self.pred_real, self.D_mask_real = self.netD(self.tgt)

        assert self.pred_real.shape[0] == self.opt.batch_size, f"prediction \
            shape incorrect ({self.pred_real.shape}, B: {self.opt.batch_size})"

        self.pred_fake, self.D_mask_fake = self.netD(self.composite)
        self.pred_anti_sc, self.D_mask_antisc = self.netD(self.anti_sc)
        if self.train_on_gf:
            self.pred_gr_fake, self.D_mask_grfake = self.netD(
                self.grounded_fake)

        # also compute the accuracy of discriminator
        if self.total_iters % self.opt.print_freq:
            B = self.opt.batch_size
            self.acc_real = len(self.pred_real[self.pred_real > 0.5]) / B
            self.acc_fake = len(self.pred_fake[self.pred_fake < 0.5]) / B
            if self.train_on_gf:
                self.acc_grfake = len(self.pred_gr_fake[self.pred_gr_fake
                    < 0.5]) / B


    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in
        every training iteration. Discriminator predictions have been computed
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
        """Calculate losses, gradients, and update network weights; called in
        every training iteration"""

        # compute the GAN losses using predictions from forward pass
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_fake = self.criterionGAN(self.pred_fake.detach(), False)
        if self.train_on_gf:
            self.loss_D_gr_fake = self.criterionGAN(self.pred_gr_fake, False)

        # compute auxiliary loss, directly use lambda for plotting purposes
        # detach all masks coming from G to prevent gradients in G
        self.loss_AUX = self.opt.lambda_aux * self.criterionMask(
            self.D_mask_real, self.D_mask_fake.detach(),
            self.D_mask_antisc.detach(), self.D_mask_grfake,
            self.g_mask.detach(), self.mask_gf, use_gf=self.train_on_gf)

        # sum the losses
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_AUX

        if self.train_on_gf:
            self.loss_D = self.loss_D + self.loss_D_gr_fake

        # Calculate gradients of discriminator
        self.loss_D.backward()


    def optimize_parameters(self):
        """Update network weights; it is called in every training iteration.
        only perform  optimizer steps after all backward operations, torch1.5
        gives an error, see https://github.com/pytorch/pytorch/issues/39141

        Arguments:
            - total_iters: training progress in steps, used to give D a
            headstart
        """

        # perform forward step
        self.forward()

        # train D and G in alternating fashion
        if self.train_G:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()



    def run_batch(self, data, total_iters):
        """
        This method incorporates the set_input and optimize_parameters
        functions, and does some checks beforehand

        """
        self.total_iters = total_iters

        # sat some boolean variables needed in threshold training curriculum
        self.headstart_over = total_iters >= self.D_headstart
        self.even_batch = (total_iters/self.opt.batch_size) % 2 == 0
        # is D performing well on the fake images generated by G?
        self.D_above_thresh = self.acc_fake > self.opt.D_threshold
        # Does D perform perfectly on the grounded fakes?
        self.D_gf_perfect = self.acc_grfake > 0.99
        # every 20 batches, run everything to update accuracies
        train_all = total_iters % (20 * self.opt.batch_size) == 0

        # by default train D (in headstart or performing below threshold:
        self.train_G = False

        # G is trained
        if self.headstart_over and self.even_batch and self.D_above_thresh:
            self.train_G = True

        # determine if grounded fakes are still used in training
        self.train_on_gf = True
        if self.D_gf_perfect:
            self.train_on_gf = False

        if total_iters == self.D_headstart:
            print("Headstart D over, starting G training")

        if train_all:
            self.train_G = self.train_on_gf = True

        # unpack data from dataset and apply preprocessing
        self.set_input(data)
        # calculate loss functions, get gradients, update network weights
        self.optimize_parameters()






