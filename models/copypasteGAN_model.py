""" CopyPasteGAN model template

You can specify '--model copypasteGAN' to use this model.

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

torch.autograd.set_detect_anomaly(True)


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
            n_epochs=1,
            n_epochs_decay=3,
            netG="copy",
            netD="copy",
            dataroot="datasets",
            save_epoch_freq=10,
            display_freq=100,
            print_freq=20,
        )

        # define new arguments for this model
        if is_train:
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
                "--nr_obj_classes",
                type=int,
                default=1,
                help="Number of object classes in images, used for multiple masks",
            )
            parser.add_argument(
                "--D_headstart",
                type=int,
                default=1000,
                help="First train only discriminator for D_headstart iterations \
                (images, independent on batchsize",
            )
            parser.add_argument(
                "--beta2",
                type=int,
                default=0.999,
                help="beta2 parameter for the adam optimizer",
            )
            parser.add_argument(
                "--sigma_blur",
                type=float,
                default=1.0,
                help="Sigma used in Gaussian filter used for blurring \
                discriminator input",
            )
            parser.add_argument(
                "--real_target",
                type=float,
                default=0.8,
                help="Target label for the discriminator, can be set <1 to \
                prevent overfitting",
            )
            parser.add_argument(
                "--seed",
                type=int,
                default=0,
                help="Provide an integer for setting the random seed. Set to \
                    0 for random seed",
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
                default=0.6,
                help="when the accuracy of the discriminator is lower than \
                    this threshold, only train D",
            )
            parser.add_argument(
                "--val_freq",
                type=int,
                default=100,
                help="every val_freq batches run the model on validation data,\
                    and obtain accuracies for training schedule.",
            )
            parser.add_argument(
                "--val_batch_size",
                type=int,
                default=128,
                help="every val_freq batches run the model on validation \
                    data, and obtain accuracies for training schedule.",
            )
            parser.add_argument(
                "--keep_last_batch",
                action="store_true",
                help="drop last incomplete batch by default",
            )
            parser.add_argument(
                "--patch_D",
                action="store_true",
                help="If true, discriminator scores individual patches on \
                    realness, else, two linear layers yield a scalar score",
            )
            parser.add_argument(
                "--accumulation_steps",
                type=int,
                default=4,
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


        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        """
        BaseModel.__init__(self, opt)

        self.multi_layered = opt.nr_obj_classes != 1
        self.D_headstart = opt.D_headstart
        self.patch = opt.patch_D

        # specify random seed
        if opt.seed == 0:
            opt.seed = np.random.randint(0, 42)
            print(f"Random seed is set to {opt.seed}")

        torch.manual_seed(opt.seed)


        # specify the training losses you want to print out.
        # base_model.get_current_losses is used for plotting and saving these
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
        if self.multi_layered:
            self.loss_names.append("loss_G_distinct")
        if not opt.no_grfakes:
            self.loss_names.extend(["loss_D_gr_fake", "acc_grfake"])
            self.train_on_gf = False
        else:
            self.train_on_gf = True

        # init all losses and accs that are used at 0 for plotting
        for loss in self.loss_names:
            setattr(self, loss, 0)

        # init for training curriculum
        self.D_gf_perfect = self.D_above_thresh = False

        # init count for gradient accumulation, simulating larger batch size
        self.count_D = self.count_G = 0

        # init gradient scaler from cuda AMP
        self.scaler = GradScaler()

        # specify the images that are saved and displayed
        # (via base_model.get_current_visuals)
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

        # define generator, output_nc is set to nr of object classes (1)
        self.netG = networks.define_G(
            opt.input_nc,
            opt.nr_obj_classes,
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
                patchGAN=opt.patch_D,
                aux=self.aux,
            )
            self.model_names.append("D")

            # define loss functions
            self.criterionGAN = networks.GANLoss(
                gan_mode="vanilla", target_real_label=opt.real_target
            ).to(self.device)
            self.criterionMask = networks.MaskLoss().to(self.device)
            self.criterionConf = networks.ConfidenceLoss().to(self.device)

            # not used at the moment
            if self.multi_layered:
                self.criterionDist = networks.DistinctMaskLoss(
                    opt.nr_obj_classes
                ).to(self.device)

            # define optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata
            information.
        """

        # put image data on device
        self.src = input["src"].to(self.device)
        self.tgt = input["tgt"].to(self.device)

        # create a grounded fake, the function samples a random polygon mask
        if self.train_on_gf and not self.opt.no_grfakes:
            self.grounded_fake, self.mask_gf = networks.composite_image(
                self.src, self.tgt, device=self.device
            )

    def forward(self, valid=False, generator=False):
        """Run forward pass. This will be called by both functions <
        optimize_parameters> and <test>.
        Parameters:
            valid: if running the validations set, anti shortcut is not
                computed, and the accuracies are computed
        """

        # generate output image given the input batch
        self.g_mask_layered = self.netG(self.src)
        self.g_mask = torch.max(self.g_mask_layered, dim=1, keepdim=True)[0]

        # set mask as attribute of loss class
        self.criterionGAN.set_copy_mask(self.g_mask)

        # binary mask for visualization
        self.g_mask_binary = networks.mask_to_binary(self.g_mask)

        # create the composite mask from src and tgt images, and predicted mask
        # this is only done when D is trained,
        self.composite, _ = networks.composite_image(
                self.src, self.tgt, self.g_mask, device=self.device
            )
            # get discriminators prediction on the generated image
        self.pred_fake_single, self.pred_fake_patch, self.D_mask_fake = self.netD(self.composite)

        # apply the masks on different source images, should be labeled false
        # we reverse the src images over the batch dimension
        if not valid:
            # use flip to "shuffle" the batch and get new combinations
            self.anti_sc_src = torch.flip(self.src, [0])
            self.anti_sc, _ = networks.composite_image(
                self.anti_sc_src, self.tgt, self.g_mask
            )
            self.pred_antisc_single, self.pred_antisc_patch, self.D_mask_antisc = self.netD(self.anti_sc)

        # get predictions from discriminators for all images (use tgt/src)
        if not generator or valid:
            self.pred_real_single, self.pred_real_patch, self.D_mask_real = self.netD(self.tgt)

        # make sure the predictions are the right size
        assert (valid or self.pred_real_single.shape[0] == self.opt.batch_size), f"prediction shape incorrect ({self.pred_real_single.shape}, B: \
            {self.opt.batch_size})"


        if (self.train_on_gf and not generator) or valid:
            self.pred_grfake_single, self.pred_grfake_patch, self.D_mask_grfake = self.netD(
                self.grounded_fake
            )

        # compute accuracy of discriminator if in validation mode
        if valid:
            self.compute_single_accs()

    def compute_accs(self):
        """
        Computes the accuracies of the discriminator

        # NB: not used at the moment, compute_single_accs() is used.
        """
        # assign the fakest patch in case of patch discriminator, else use the scaler prediction
        patch = self.pred_real_pathc.dim() > 2
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

    def compute_single_accs(self):
        B = self.opt.val_batch_size

        self.acc_real = len(self.pred_real_single[self.pred_real_single>0.5])/B
        self.acc_fake = len(self.pred_fake_single[self.pred_fake_single<0.5])/B

        if self.train_on_gf:
            self.acc_grfake = (
                len(self.pred_grfake_single[self.pred_grfake_single < 0.5]) / B
            )





    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in
        every training iteration. Discriminator predictions have been computed
        in the backward_D"""

        # compute generator losses
        self.loss_G_comp = self.criterionGAN(self.pred_fake_patch, True, self.patch)
        self.loss_G_anti_sc = self.criterionGAN(self.pred_antisc_patch, False, self.patch)
        self.loss_G_conf = (
            self.opt.confidence_weight * self.criterionConf(self.g_mask)
            if self.opt.confidence_weight > 0
            else 0
        )

        # sum components
        self.loss_G = self.loss_G_comp + self.loss_G_anti_sc + self.loss_G_conf

        # not used atm
        if self.multi_layered:
            self.loss_G_distinct = self.criterionDist(self.g_mask_layered)
            self.loss_G = self.loss_G + self.loss_G_distinct

        # scale the loss and perform backward step
        self.scaler.scale(self.loss_G / self.opt.accumulation_steps).backward()

    def backward_D(self):
        """Calculate losses, gradients, and update network weights; called in
        every training iteration"""

        # compute the GAN losses using predictions from forward pass
        # real is not computed using patch, as all patches are real
        print(type(self.pred_real_patch))
        self.loss_D_real = self.criterionGAN(self.pred_real_patch, True, False)
        self.loss_D_fake = self.criterionGAN(self.pred_fake_patch.detach(), False, self.patch)
        if self.train_on_gf:
            self.loss_D_gr_fake = self.criterionGAN(self.pred_grfake_patch, False, self.patch)

        # compute auxiliary loss, directly use lambda for plotting purposes
        # detach all masks coming from G to prevent gradients in G
        self.loss_AUX = (
            self.opt.lambda_aux
            * self.criterionMask(
                self.D_mask_real,
                self.D_mask_fake.detach(),
                self.D_mask_antisc.detach(),
                self.D_mask_grfake,
                self.g_mask.detach(),
                self.mask_gf,
                use_gf=self.train_on_gf,
            )
            if self.aux > 0
            else 0
        )

        # sum the losses
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_AUX

        if self.train_on_gf:
            self.loss_D = self.loss_D + self.loss_D_gr_fake

        # scale gradients and perform backward step
        self.scaler.scale(self.loss_D / self.opt.accumulation_steps).backward()

    def optimize_parameters(self):
        """Update network weights; it is called in every training iteration.
        only perform  optimizer steps after all backward operations, torch1.5
        gives an error, see https://github.com/pytorch/pytorch/issues/39141
        """

        # perform forward step
        self.forward(generator=self.train_G)

        # perform gradient accumulation to simulate larger batch size
        # for more information, see: https://towardsdatascience.com/i-am-so-done-with-cuda-out-of-memory-c62f42947dca
        update_freq = self.opt.accumulation_steps

        # either train G or D, using the AMP scaler
        if self.train_G:
            self.backward_G()
            self.count_G += 1
            if self.count_G % update_freq == 0:
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
                self.optimizer_G.zero_grad()
        else:
            self.backward_D()
            self.count_D += 1
            if self.count_D % update_freq == 0:
                self.scaler.step(self.optimizer_D)
                self.scaler.update()
                self.optimizer_D.zero_grad()

    def run_batch(self, data, total_batches):
        """
        This method incorporates the set_input and optimize_parameters
        functions, and does some checks beforehand

        Arguments:
            - total_iters: training progress in steps, used to give D a
            headstart

        """

        # tracemalloc.start()
        if total_batches == self.D_headstart:
            print("Headstart D over")

        # sat some boolean variables needed in threshold training curriculum
        self.headstart_over = total_batches >= self.D_headstart
        self.even_batch = total_batches % 2 == 0

        # by default train D (in headstart or performing below threshold:
        self.train_G = False

        # G is trained
        if self.headstart_over and self.even_batch and self.D_above_thresh:
            self.train_G = True

        # determine if grounded fakes are still used in training
        if self.D_gf_perfect and self.headstart_over:
            self.train_on_gf = False

        # unpack data from dataset and apply preprocessing
        self.set_input(data)
        # calculate loss functions, get gradients, update network weights
        self.optimize_parameters()

        # "train memory"
        # snapshot = tracemalloc.take_snapshot()
        # self.print_snapshot(snapshot.statistics('lineno'))

    def print_snapshot(self, top_stats):
        for index, stat in enumerate(top_stats[:3], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#%s: %s:%s: %.1f KiB"
                  % (index, filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

    def run_validation(self, val_data):
        """
        Run the complete validation set, and set booleans describing the model
        performance. These are used to determine the training schedule.
        """

        # tracemalloc.start()

        # reset all conditional parameters
        self.train_on_gf = not self.opt.no_grfakes
        self.D_above_thresh = False
        self.D_gf_perfect = False

        # init average lists
        acc_gf = []
        acc_real = []
        acc_fake = []

        # compute accuracy on the validation data
        with torch.no_grad():
            for i, data in enumerate(val_data):
                self.set_input(data)
                with autocast():
                    self.forward(valid=True)

                # save accuracies
                acc_gf.append(self.acc_grfake)
                acc_fake.append(self.acc_fake)
                acc_real.append(self.acc_real)

        # set accuracies to mean for plotting purposes
        self.acc_grfake = np.mean(acc_gf)
        self.acc_fake = np.mean(acc_fake)
        self.acc_real = np.mean(acc_real)

        # determine training curriculum for next session
        # performance of discriminator on grounded fakes
        self.D_gf_perfect = self.acc_grfake > 0.99

        # check performance on fakes to determine whether to train G
        self.D_above_thresh = self.acc_fake > self.opt.D_threshold

        # print validation scores
        if self.opt.verbose:
            print(
                f"validation accuracies:\n\
                gf: {self.acc_grfake:.2f}\n\
                real: {self.acc_real:.2f}\n\
                fake: {self.acc_fake:.2f}\n"
            )
        # snapshot = tracemalloc.take_snapshot()

        # print("validation memory")
        # self.print_snapshot(snapshot.statistics('lineno'))








