""" CopyGAN model

You can specify '--model copy' to use this model.

Author: Tom Lotze

"""


import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# imports for memory management
import tracemalloc
import linecache
import os


from models.base_model import BaseModel
import models.networks as networks
from util import util

# for testing
from kornia.morphology import erosion, dilation
from scipy import ndimage



class CopyModel(BaseModel):
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

        # set default options for this model
        parser.set_defaults(
            dataset_mode="double",
            name="CopyGAN",
            load_size=70,
            crop_size=64,
            batch_size=64,
            lr=2e-4,
            lr_policy="step",
            direction=None,
            n_epochs=20,
            n_epochs_decay=10,
            netG="copy",
            netD="copy",
            dataroot="datasets",
            save_epoch_freq=10,
            display_freq=100,
            print_freq=20,
            real_target=0.9,
            fake_target=0.1
        )

        # define new arguments for this model
        # if is_train:
        parser.add_argument(
            "--lambda_aux",
            type=float,
            default=0.1,
            help="weight for the auxiliary mask loss",
        )
        parser.add_argument(
            "--confidence_weight",
            type=float,
            default=0.0,
            help="weight for the confidence loss for generator",
        )
        parser.add_argument(
            "--D_headstart",
            type=int,
            default=0,
            help="First train only discriminator for D_headstart batches",
        )
        parser.add_argument(
            "--sigma_blur",
            type=float,
            default=1.0,
            help="Sigma used in Gaussian filter used for blurring \
            discriminator input",
        )
        parser.add_argument(
            "--no_border_zeroing",
            action="store_true",
            help="default: clamp borders of generated mask to 0 \
                (store_false)",
        )
        parser.add_argument(
            "--D_threshold",
            type=float,
            default=0.5,
            help="when the accuracy of the discriminator is lower than \
                this threshold, only train D",
        )
        parser.add_argument(
            "--pool_D",
            action="store_true",
            help="If true, discriminator uses avg pooling to get its prediction, like Arandjelovic, otherwise conv & linear layers are used",
        )
        parser.add_argument(
            "--accumulation_steps",
            type=int,
            default=1,
            help="accumulate gradients for this amount of batches, \
                before backpropagating, to simulate a larger batch size",
        )
        parser.add_argument(
            "--no_grfakes",
            action="store_true",
            help="If true, no grounded fakes will be used in training",
        )
        parser.add_argument(
            "--flip_vertical",
            action="store_true",
            help="If specified, the data will be flipped vertically",
        )
        parser.add_argument(
            "--sequential",
            action="store_true",
            help="If specified, G and D will not be trained in alternating\
                fashion, but sequentially (for val_freq batches each)",
        )
        parser.add_argument(
            "--noisy_labels",
            action="store_true",
            help="If specified, random noise will be added to the target labels in the adversarial loss",
        )
        parser.add_argument(
            "--use_amp",
            action="store_true",
            help="If specified, gradient scaling using AMP GradScaler is enabled",
        )

        return parser


    def __init__(self, opt):
        """Initialize this model class.

        Parameters: cal
            opt -- training/test options

        """
        BaseModel.__init__(self, opt)

        self.D_headstart = opt.D_headstart
        self.pool = opt.pool_D

        # specify random seed
        if opt.seed == 0:
            opt.seed = np.random.randint(0, 42)
            print(f"Random seed is set to {opt.seed}")

        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        # specify the training losses to print (base_model.get_current_losses)
        self.loss_names = [
            "loss_G_comp",
            "loss_G_anti_sc",
            "loss_G",
            "loss_D_real",
            "loss_D_fake",
            "loss_D",
            "acc_real",
            "acc_fake",
        ]

        # if the model is not copy, we cannot use the auxiliary loss
        if opt.netD != "copy" and opt.lambda_aux > 0:
            print(
                f"CopyDiscriminator not used, auxiliary weight set to 0 \
                (instead of {opt.lambda_aux})"
            )
            opt.lambda_aux = 0

        # make sure invalid combination cannot be used
        self.aux = opt.lambda_aux > 0
        if opt.no_grfakes and self.aux:
            raise Exception(
                "invalid options combination. If grounded fakes are not used,\
                 auxiliary loss cannot be used either.\n Exiting..."
            )

        # add other losses if specified
        if opt.confidence_weight > 0:
            self.loss_names.append("loss_G_conf")
        if self.aux:
            self.loss_names.append("loss_AUX")
        if not opt.no_grfakes:
            self.loss_names.extend(["loss_D_gr_fake", "acc_grfake"])
            self.train_on_gf = False
        else:
            self.train_on_gf = True

        # init all losses and accs that are used at 0 for plotting
        for loss in self.loss_names:
            setattr(self, loss, 0)

        # init for training curriculum
        self.D_gf_perfect, self.D_above_thresh = False, False

        # keep count of training steps (also for gradient accumulation)
        self.count_D, self.count_G = 0, 0

        # init gradient scaler from cuda AMP
        self.scaler = GradScaler(enabled=opt.use_amp)

        # specify the images that are saved and displayed
        # (via base_model.get_current_visuals)
        if not self.isTrain:
            self.visual_names =[
                    "src",
                    "tgt",
                    "g_mask",
                    "g_mask_binary",
                    "ground_truth",
                    "eroded_mask",
                    "composite_eroded",
                    "composite",
                    "labelled_mask"]
        else:
            if self.aux:
                self.visual_names = [
                    "src",
                    "tgt",
                    "g_mask",
                    "g_mask_binary",
                    "composite",
                    "D_mask_fake",
                    "anti_sc_src",
                    "anti_sc",
                    "D_mask_antisc",
                    "D_mask_real",
                ]
                if not opt.no_grfakes:
                    self.visual_names.extend(
                        ["grounded_fake", "D_mask_grfake", "mask_gf"]
                    )
            else:
                self.visual_names = [
                    "src",
                    "tgt",
                    "g_mask",
                    "g_mask_binary",
                    "composite",
                    "anti_sc_src",
                    "anti_sc",
                ]
                if not opt.no_grfakes:
                    self.visual_names.extend(["grounded_fake", "mask_gf"])

        # define generator, output_nc is set to 1 (binary mask)
        self.netG = networks.define_G(
            opt.input_nc,
            1,
            ngf=opt.ngf,
            netG=opt.netG,
            norm=opt.norm,
            border_zeroing=not opt.no_border_zeroing,
            gpu_ids=self.gpu_ids,
            img_dim=opt.crop_size,
        )

        # G must be saved to disk
        self.model_names = ["G"]

        if self.isTrain:
            # only define the discriminator if in training phase
            self.netD = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                norm=opt.norm,
                gpu_ids=self.gpu_ids,
                img_dim=opt.crop_size,
                sigma_blur=opt.sigma_blur,
                pool=opt.pool_D,
                aux=self.aux,
            )
            self.model_names.append("D")

            # define loss functions
            self.criterionGAN = networks.GANLoss(
                gan_mode="vanilla", target_real_label=opt.real_target,
                target_fake_label=opt.fake_target, noisy_labels=
                opt.noisy_labels).to(self.device)
            self.criterionMask = networks.MaskLoss().to(self.device)
            self.criterionConf = networks.ConfidenceLoss().to(self.device)

            # define optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]


    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata
            information.
        """

        # put image data on device
        self.src = input["src"].to(self.device)
        self.tgt = input["tgt"].to(self.device)
        self.irrel = input['irrel'].to(self.device)

        if not self.isTrain:
            self.ground_truth = input['gt']
            self.bin_gt = util.mask_to_binary((self.ground_truth[:, -1]+1)/2)

        # create a grounded fake, the function samples a random polygon mask
        if self.train_on_gf and not self.opt.no_grfakes:
            self.grounded_fake, self.mask_gf = networks.composite_image(
                self.src, self.tgt, device=self.device
            )


    def forward(self, valid=False, generator=False):
        """Run forward pass. This will be called by functions
        <optimize_parameters>, <test>, and <run_validation>
        Parameters:
            valid: if running the validations set, anti shortcut is not
                computed, and the accuracies are computed
            generator: used to determine which predictions must be computed
        """

        # generate output image given the input batch
        self.g_mask = self.netG(self.src)
        # binary mask for visualization
        self.g_mask_binary = util.mask_to_binary(self.g_mask)

        # create the composite mask from src and tgt images, and predicted mask
        self.composite, _ = networks.composite_image(self.src, self.tgt,
            self.g_mask, device=self.device)

        # # if we are training D, prevent gradient flow back through G
        if not generator:
            self.composite = self.composite.detach()

        # get discriminators prediction on the generated (fake) image
        self.pred_fake, self.D_mask_fake = self.netD(self.composite)

        # apply the masks on different source images: anti shortcut images
        if not valid:
            # use flip to "shuffle" the batch and get new combinations
            # self.anti_sc_src = torch.flip(self.src, [0])
            # self.anti_sc, _ = networks.composite_image(self.anti_sc_src, self.tgt, self.g_mask)

            # use an irrelant image instead of shuffled src images
            self.anti_sc, _ = networks.composite_image(self.irrel, self.tgt, self.g_mask)

            if not generator:
                self.anti_sc = self.anti_sc.detach()

            self.pred_antisc, self.D_mask_antisc= self.netD(self.anti_sc)

        # get predictions from discriminators for real images (use tgt/src)
        if not generator or valid:
            self.pred_real, self.D_mask_real = self.netD(self.tgt)

        # compute grounded fake predictions
        if (self.train_on_gf and not generator) or valid:
            self.pred_grfake,self.D_mask_grfake = self.netD(self.grounded_fake)

        # compute accuracy of discriminator if in validation mode
        if valid:
            self.compute_accs()


    def compute_accs(self):
        B = self.opt.val_batch_size
        # use 0.5 as cutoff value for accuracy
        self.acc_real = len(self.pred_real[self.pred_real>0.5])/B
        self.acc_fake = len(self.pred_fake[self.pred_fake<0.5])/B

        if self.train_on_gf:
            self.acc_grfake = len(self.pred_grfake[self.pred_grfake < 0.5])/B


    def backward_G(self):
        """
        Calculate losses and gradients for Generator
        Discriminator predictions have been computed in the forward pass"""

        # compute adversarial losses
        self.loss_G_comp = self.criterionGAN(self.pred_fake, True)
        self.loss_G_anti_sc = self.criterionGAN(self.pred_antisc, False)
        # compute confidence loss if used
        self.loss_G_conf = (
            self.opt.confidence_weight * self.criterionConf(self.g_mask)
            if self.opt.confidence_weight > 0
            else 0
        )

        # sum components
        self.loss_G = self.loss_G_comp + self.loss_G_anti_sc + self.loss_G_conf

        # scale the loss and perform backward step
        # self.scaler.scale(self.loss_G).backward()
        self.loss_G.backward()


    def backward_D(self):
        """Calculate losses and gradients for Disciminator"""

        # compute adversarial losses
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        if self.train_on_gf:
            self.loss_D_gr_fake = self.criterionGAN(self.pred_grfake, False)

        # compute auxiliary loss, directly use lambda for plotting purposes
        # detach all masks coming from G to prevent gradients in G
        self.loss_AUX = (
            self.opt.lambda_aux * self.criterionMask(
                self.D_mask_real,
                self.D_mask_fake,
                self.D_mask_antisc,
                self.D_mask_grfake,
                self.g_mask.detach(),
                self.mask_gf, # does not require grads, no need to detach
                use_gf=self.train_on_gf
            )
            if self.aux > 0
            else 0)

        # sum the components to yield total loss
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_AUX
        if self.train_on_gf:
            self.loss_D = self.loss_D + self.loss_D_gr_fake

        # scale gradients and perform backward step
        # self.scaler.scale(self.loss_D).backward()
        self.loss_D.backward()


    def optimize_parameters(self):
        """
        Update network weights for either Generator or Discriminator
        """

        # for testing purposes,always 1/1 after headstart
        self.train_G = self.even_batch and self.headstart_over

        # perform forward step
        self.forward(generator=self.train_G)

        # either train G or D, using the AMP scaler
        if self.train_G:
            self.count_G += 1
            self.optimizer_G.zero_grad()
            self.backward_G()
            if self.total_batches % self.opt.print_freq == 0: util.print_gradients(self.netG)

            self.optimizer_G.step()
            # self.scaler.step(self.optimizer_G)
            # self.scaler.update()
            # self.optimizer_G.zero_grad()

        else:
            self.count_D += 1
            self.optimizer_D.zero_grad()
            self.backward_D()
            if (self.total_batches+1) % self.opt.print_freq == 0: util.print_gradients(self.netD)
            self.optimizer_D.step()

            # self.scaler.step(self.optimizer_D)
            # self.scaler.update()
            # self.optimizer_D.zero_grad()



    def run_batch(self, data, total_batches):
        """
        This method wraps the set_input and optimize_parameters
        functions, and does some checks to determine training curriculum
        Most of the booleans used for performance checks are set in
        <run_validation>

        Arguments:
            - data: current training batch
            - total_batches: training progress in batches
        """

        if self.opt.tracemalloc:
            tracemalloc.start()

        self.total_batches = total_batches

        if total_batches == self.D_headstart:
            print("Headstart D over")

        # determine training progress and curriculum
        self.headstart_over = total_batches > self.D_headstart
        self.even_batch = total_batches % 2 == 0

        # by default train D (in headstart or performing below threshold:
        self.train_G = False

        # determine if G can be trained
        # G and D are trained sequentially for eval_freq batches
        alt_cond = self.opt.sequential and ((total_batches // 100) % 2 == 0)
        batch_right = self.even_batch or alt_cond

        if self.headstart_over and self.D_above_thresh and batch_right:
            self.train_G = True

        # determine if grounded fakes are still used in training
        if self.D_gf_perfect and self.headstart_over:
            self.train_on_gf = False

        # unpack data from dataset and apply preprocessing
        self.set_input(data)
        # calculate loss functions, get gradients, update network weights

        self.optimize_parameters()

        if self.opt.tracemalloc:
            snapshot = tracemalloc.take_snapshot()
            print("Tracemalloc: training memory")
            util.print_snapshot(snapshot)


    def run_validation(self, val_data):
        """
        Run the complete validation set, and set booleans describing the model
        performance. These are used to determine the training schedule.

        Arguments:
            - val_data: batch of validation data
        """

        if self.opt.tracemalloc:
            tracemalloc.start()

        # reset all conditional parameters
        self.train_on_gf = not self.opt.no_grfakes
        self.D_above_thresh = False
        self.D_gf_perfect = False

        # init average lists
        acc_gf, acc_real, acc_fake = [], [], []
        preds_grfake, preds_fake, preds_real = [], [], []

        # compute accuracy on the validation data
        with torch.no_grad():
            for i, data in enumerate(val_data):
                # preprocess data and perform forward pass
                self.set_input(data)
                with autocast():
                    self.forward(valid=True)

                # save accuracies
                acc_gf.append(self.acc_grfake)
                acc_fake.append(self.acc_fake)
                acc_real.append(self.acc_real)

                preds_grfake.append(self.pred_grfake.mean().item())
                preds_fake.append(self.pred_fake.mean().item())
                preds_real.append(self.pred_real.mean().item())


        # set accuracies to mean for plotting purposes
        self.acc_grfake = np.mean(acc_gf)
        self.acc_fake = np.mean(acc_fake)
        self.acc_real = np.mean(acc_real)

        # set all training curriculum booleans for the coming eval_freq batches
        # performance of discriminator on grounded fakes
        self.D_gf_perfect = self.acc_grfake > 0.99

        # check performance on fakes to determine whether to train G
        self.D_above_thresh = self.acc_fake > self.opt.D_threshold

        # print validation scores
        if self.opt.verbose:
            print(
                f"validation accuracies:\n\
                gf: {self.acc_grfake:.2f}, {np.mean(preds_grfake)}\n\
                real: {self.acc_real:.2f},  {np.mean(preds_real)}\n\
                fake: {self.acc_fake:.2f}, {np.mean(preds_fake)}\n"
            )

        if self.opt.tracemalloc:
            snapshot = tracemalloc.take_snapshot()
            print("Tracemalloc: validation memory")
            util.print_snapshot(snapshot)


    def test(self, data):
        assert not self.isTrain, "Model should be in testing state"

        with torch.no_grad():
            self.set_input(data)

            self.g_mask = self.netG(self.src)

            # binary mask for visualization
            self.g_mask_binary = util.mask_to_binary(self.g_mask)

            # try erosion to remove artefacts
            # kernel = torch.ones((3,3))
            # kernel = torch.Tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

            kernel_plus = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

            kernel_ones = torch.ones(3, 3)
            self.eroded_mask = dilation(erosion(self.g_mask_binary, kernel_plus), kernel_ones)


            self.labelled_mask, num_mask = ndimage.label(self.eroded_mask)

            self.labelled_mask = torch.Tensor(self.labelled_mask)
            self.labelled_mask= (self.labelled_mask / self.labelled_mask.max())

            self.composite, _ = networks.composite_image(
                    self.src, self.tgt, self.g_mask_binary, device=self.device)

            self.composite_eroded, _ = networks.composite_image(
                    self.src, self.tgt, self.eroded_mask, device=self.device)











