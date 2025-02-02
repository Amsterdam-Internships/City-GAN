import torch
import numpy as np
from .base_model import BaseModel
from util import util
from . import networks

from torchvision.transforms.functional import affine
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import random
import torch.nn.functional as F
from math import log


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
        parser.set_defaults(dataset_mode='room', preprocess="resize", load_size=64, crop_size=64, no_flip=True, netD='basic', init_type="normal", name="MoveModel", lr_policy="step", gan_mode="vanilla", real_target=0.9, fake_target=0.1)  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        parser.add_argument('--use_eq_loss', action='store_true', help='If specified, the equality loss will be used, penalizing little difference with the original image')
        parser.add_argument('--theta_dim', type=int, default=2, choices=[2, 6], help= "specify how many params to use for the affine tranformation. Either 6 (full theta) or 2 (translation only)")
        parser.add_argument('--n_layers_conv', type=int, default=4, help='used for convnet in move model')
        parser.add_argument('--two_stream', action='store_true', help='If True, the object and target will separately go through a layer block before being concatenated, instead of concatenated beforehand and fed to the same layer directly')




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

        # specify random seed
        if opt.seed == 0:
            opt.seed = np.random.randint(0, 42)
            print(f"Random seed is set to {opt.seed}")

        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        self.loss_names = ["loss_D_real", "loss_D_fake",  "loss_D", "loss_G", "loss_conv", "acc_real", "acc_fake"]
        if opt.use_eq_loss:
            self.loss_names.append("loss_eq")

        # for sanity checking
        # self.loss_names = ["loss_G"]

        for loss in self.loss_names:
            setattr(self, loss, 0)

        # define variables for plotting and saving
        self.visual_names = ["tgt", "src", "mask_binary", "obj", "transf_obj", "composite"]

        for v in self.visual_names:
            setattr(self, v, torch.ones(1, 1, 64, 64))

        n_layers = -3 + int(log(opt.crop_size, 2))


        # for sanity checking
        # self.visual_names =  ["tgt", "src", "mask_binary", "obj", "transf_obj_mask", "GT"]

        self.count_G, self.count_D = 0, 0

        # define the convnet that predicts theta
        # perhaps we should treat the object and target separately first
        conv_input_nc = 3 if opt.two_stream else 6

        self.netConv = networks.define_D(conv_input_nc, opt.ngf, netD="move", n_layers_D=n_layers, gpu_ids=self.gpu_ids, norm=opt.norm, init_type=opt.init_type, two_stream=opt.two_stream, img_dim=opt.crop_size)


        self.model_names = ["Conv"]

        if self.isTrain:
            # define Discriminator
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, gpu_ids=self.gpu_ids, n_layers_D=n_layers)
            self.model_names.append("D")

            self.scaler = GradScaler(enabled=opt.use_amp)

            if opt.tracemalloc:
                import tracemalloc

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, target_real_label= opt.real_target, target_fake_label=opt.fake_target, noisy_labels=opt.noisy_labels).to(self.device)
            self.MSE = torch.nn.MSELoss(reduction='none')

            # blur to apply on input to discriminator
            self.blur = networks.GaussianSmoothing(sigma=(0.5, 0.5)).to(self.device)

            # for sanity checking
            # self.theta_gt_single = torch.Tensor([[1.2, 0, -0.5],[0, 0.8, 0.3]]).to(self.device)
            # self.pred_fake = torch.tensor([0])


            # define optimizers
            self.optimizer_Conv = torch.optim.Adam(
                self.netConv.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))
            self.optimizers = [self.optimizer_Conv, self.optimizer_D]
            # self.optimizers = [self.optimizer_Conv]


        # The program will automatically call <model.setup> to define schedulers, load networks, and print networks



    def center_object(self):
        """
        This function performs the following steps:
        - Mask is used to extract the object from the source image
        - The object and corresponding mask are centered
        """

        # get the surface per mask in the batch
        self.surface = self.mask_binary.sum([2, 3]).view(-1, 1, 1, 1)
        # center and return the object
        # inspired on https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
        mask_pdist = self.mask_binary / self.surface

        [self.B, _, self.w, self.h] = list(mask_pdist.shape)
        # assert B == self.opt.batch_size, f"incorrect batch dim: {B}"

        self.x_center, self.y_center = self.w//2, self.h//2

        # marginal distributions
        dx = torch.sum(mask_pdist, 2).view(-1, self.w)
        dy = torch.sum(mask_pdist, 3).view(-1, self.h)

        # expected values
        self.cx = torch.sum(dx * torch.arange(self.w).to(self.device), 1)
        self.cy = torch.sum(dy * torch.arange(self.h).to(self.device), 1)

        # compute necessary translations
        x_t = self.x_center - self.cx
        y_t = self.y_center - self.cy

        # extract object from src
        obj = (self.mask_binary) * self.src

        # translate the object and mask to center
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
        self.obj, self.obj_mask = self.center_object()



    def forward(self, valid=False, generator=False, baseline=False):
        """
        what needs to be done:
            - target image and centered object are fed to convnet (initialized in the init)
            - theta parameters are the output
            - affine transformation on the object and the object mask
            - the transformed object and object masks are composited --> output img
        """
        # compute theta using the convolutional network

        # two-stream input
        if self.opt.two_stream:
            zero_centered,scale,translation = self.netConv(self.obj, self.tgt)
        else:
            # single stream input
            # concatenate the target and object on channel dimension
            tgt_obj_cat = torch.cat([self.tgt, self.obj], 1)
            zero_centered,scale,translation = self.netConv(tgt_obj_cat)

        B = translation.shape[0]

        # initialize theta
        theta = torch.zeros(B, 2, 2).to(self.device)
        # set the diagonal of theta
        theta[:, torch.eye(2).bool()] = scale.float()
        # concatenate the translation parameters
        self.theta_complete = torch.cat((theta, translation.unsqueeze(2)), 2)

        # set the other two parameters, constrain to zero for now
        # self.theta[:, 0, 1] = zero_centered[:, 0]
        # self.theta[:, 1, 0] = zero_centered[:, 1]


        # testing with theta
        # first do scaling, translation, then scaling, or opposite
        # theta_complete = torch.Tensor([[[1.5, 0, 1], [0, 0.5, 1]]])
        # theta_scale = torch.Tensor([[[1.5, 0, 0], [0, 0.5, 0]]])
        # theta_translate = torch.Tensor([[[1, 0, 1], [0, 1, 1]]])
        # self.theta_complete[0] = torch.Tensor([[1.5, 0, 1], [0, 0.5, 1]])


        # preset affine settings
        align_corners = True
        # set to zero because of weird artefacts when set to "border"
        pad = "zeros"

        # 1) APPLY SCALING FACTOR
        theta_scale = torch.zeros(B, 2, 2).to(self.device)
        # set the diagonal of theta to be the scale
        theta_scale[:, torch.eye(2).bool()] = scale.float()
        # add translation params as zeros
        self.theta_scale = torch.cat((theta_scale, torch.zeros(B, 2, 1).to(self.device)), 2)
        # compute the flow field (grid) and execute scaling on object and mask
        grid_scale = F.affine_grid(self.theta_scale, self.obj.size(), align_corners=align_corners).float()
        self.scaled_obj = F.grid_sample(self.obj, grid_scale, align_corners=align_corners, padding_mode=pad)
        self.scaled_obj_mask = F.grid_sample(self.obj_mask.float(), grid_scale, align_corners=align_corners, padding_mode=pad)


        # 2) APPLY TRANSLATION
        theta_translate = torch.zeros(B, 2, 2).to(self.device)
        # set the diagonal of theta to be the scale
        theta_translate[:, torch.eye(2).bool()] = 1
        self.theta_translate = torch.cat((theta_translate, translation.unsqueeze(2)), 2)
        # compute the flow field and execute translation on object and mask
        grid_translate = F.affine_grid(self.theta_translate, self.obj.size(), align_corners=align_corners).float()
        self.transf_obj = F.grid_sample(self.scaled_obj, grid_translate, align_corners=align_corners, padding_mode=pad)
        self.transf_obj_mask = F.grid_sample(self.scaled_obj_mask.float(), grid_translate, align_corners=align_corners, padding_mode=pad)


        # get the surfaces of the transformed objects
        if self.opt.use_eq_loss:
            self.trans_obj_surface = self.transf_obj_mask.sum((1, 2, 3))

        ############### SANITY CHECKING USING GT theta

        # self.theta_gt = self.theta_gt_single.expand(B, 2, 3)

        # grid_gt = F.affine_grid(self.theta_gt, self.obj.size(), align_corners=False).float()
        # # self.transf_obj = F.grid_sample(self.obj, grid, align_corners=False)
        # self.GT = F.grid_sample(self.obj_mask.float(), grid_gt, align_corners=False)
        ###############

        # composite the moved object with the background from the target
        self.composite, _ = networks.composite_image(self.transf_obj, self.tgt, self.transf_obj_mask)

        # detach composite if we are training the discriminator
        if not generator:
            self.composite = self.composite.detach()

        # get the prediction on the fake image, and blur the image
        if not baseline:
            self.pred_fake = self.netD(self.blur(self.composite))[0]

        if valid or not generator:
            self.pred_real = self.netD(self.blur(self.src))[0]
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

        # get the prediction on the real image: Now done in forward pass
        # self.pred_real = self.netD(self.src)

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
        self.loss_G = self.criterionGAN(self.pred_fake, True)
        # for sanity checking
        # self.loss_G = self.MSE(self.GT, self.transf_obj_mask)

        # use the inverse MSE loss to enforce the object to be in the image
        # perhaps we should scale this based on the object surface

        #TODO: Sum nr of pixels in transf max
        if self.opt.use_eq_loss:
            MSE_loss = torch.mean(self.MSE(self.composite, self.tgt), (1, 2, 3))
            # correct for transformed object surface
            size_correction = self.trans_obj_surface/self.opt.load_size**2

            ####### this was the code used for run 9
            # MSE_loss = torch.mean(self.MSE(self.composite, self.tgt), (1, 2, 3))
            # size_correction = self.surface.squeeze()/self.opt.load_size**2
            self.loss_eq = 2 - torch.mean(MSE_loss/size_correction)

            self.loss_conv = self.loss_G + self.loss_eq
        else:
            self.loss_conv = self.loss_G

        # self.scaler.scale(self.loss_G).backward()
        self.loss_conv.backward()


    def optimize_parameters(self, data):
        """Update network weights; it will be called in every training iteration.
        """


        ########## FORWARD PASS RUN 9 (NO SCALER)
        self.set_input(data)

        train_G = not(self.overall_batch % 3 == 0)
        # train_G = True

        # run the forward pass
        self.forward(generator=train_G)

        # train convnet predicting theta
        if train_G:
            # print("Training Convnet")
            self.optimizer_Conv.zero_grad()
            self.backward_Conv()

            # self.scaler.step(self.optimizer_Conv)
            # self.scaler.update()
            self.optimizer_Conv.step()
            self.count_G += 1

        # train discriminator
        else:
            # print("Training D")
            self.optimizer_D.zero_grad()
            self.backward_D()
            # self.scaler.step(self.optimizer_D)
            # self.scaler.update()
            self.optimizer_D.step()
            self.count_D += 1

        ##############

        # self.set_input(data)

        # train_G = not(self.overall_batch % 3 == 0)
        # # train_G = True

        # # run the forward pass
        # self.forward(generator=train_G)

        # # train convnet predicting theta
        # if train_G:
        #     # print("Training Convnet")

        #     self.backward_Conv()

        #     self.scaler.step(self.optimizer_Conv)
        #     self.scaler.update()
        #     # self.optimizer_Conv.step()
        #     self.count_G += 1
        #     self.optimizer_Conv.zero_grad()

        # # train discriminator
        # else:
        #     # print("Training D")

        #     self.backward_D()
        #     self.scaler.step(self.optimizer_D)
        #     self.scaler.update()
        #     # self.optimizer_D.step()
        #     self.count_D += 1
        #     self.optimizer_D.zero_grad()


    def baseline(self, data, type_='random'):

        assert type_ in {"random", "scanline", "move", "real"}, f"Type {type_} not recognized, choose from \"random\", \"scanline\", \"move\" or \"real\""

        assert self.opt.batch_size == 1,"for baselines, batch size should be 1"

        self.set_input(data)

        if type_=="move":
            # call forward pass without computing
            self.forward(generator=True, baseline=True)
            return self.src, self.composite
        elif type_ == "real":
            return self.src, self.src

        img_width, img_height = self.src.shape[2:4]

        obj_width = int(torch.max(torch.sum(self.obj_mask>0, axis=2)))
        obj_height = int(torch.max(torch.sum(self.obj_mask>0, axis=3)))

        # # background includes the object to be moved
        background = (1-self.mask_binary) * self.src
        # first we separated the object ourselves, but we can also center and then move, but we would need the initial y-coordinate in case of scanline translation
        # obj = self.mask_binary * self.src
        obj, obj_mask = self.center_object()

        # x translation is always used
        # x_translation = np.random.randint(obj_width, img_width - obj_width)
        x_translation = np.random.randint(-self.x_center, self.x_center)
        # x_translation = np.random.normal(obj_width, 2*obj_width)

        if type_=="random":
            y_translation = np.random.randint(-self.y_center, self.y_center)

        elif type_=="scanline":
            # we want to obtain x_min and x_max of the object, and the width of the object (x_max-x_min). Then we move the object to the right with at least width, and maximally (img_width - width), and modulo the transformation with img_width

            if obj_width >= img_width:
                print("object is too large, to be implemented (returns None)")
                return None, None

            # correct from center, such that object is at original y location
            y_translation = int((self.cy - self.y_center).item())

        moved_obj = affine(obj, 0, [x_translation, y_translation], 1, 0)
        new_background = 1 - (moved_obj != 0).int()
        self.moved = new_background  * self.tgt + moved_obj

        # print(x_translation, y_translation)
        # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        # ax1.imshow(util.tensor2im(self.src), origin="upper")
        # ax1.set_title("source")
        # ax2.imshow(util.tensor2im(self.tgt), origin="upper")
        # ax2.set_title("target")
        # ax3.imshow(util.tensor2im(moved_obj), origin="upper")
        # ax3.set_title(f"moved_obj ({x_translation}, {y_translation})")
        # ax4.imshow(util.tensor2im(self.moved), origin="upper")
        # ax4.set_title("result")
        # ax5.imshow(util.tensor2im(obj), origin="upper")
        # ax5.set_title("object")
        # ax6.imshow(util.tensor2im(new_background), origin="upper")
        # ax6.set_title("background")
        # plt.tight_layout()
        # plt.show()


        return self.tgt, self.moved


    def run_batch(self, data, overall_batch):
        """
        Wrapper function for optimize_parameters, general compatibility
        """
        self.overall_batch = overall_batch
        self.optimize_parameters(data)



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
        acc_real, acc_fake = [], []
        preds_real, preds_fake = [], []

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

                preds_real.append(torch.mean(self.pred_real).item())
                preds_fake.append(torch.mean(self.pred_fake).item())

        # set accuracies to mean for plotting purposes
        self.acc_fake = np.mean(acc_fake)
        self.acc_real = np.mean(acc_real)


        # print validation scores
        if self.opt.verbose:
            print(
                f"validation accuracies:\n\
                real: {self.acc_real:.2f}, {np.mean(preds_real)}\n\
                fake: {self.acc_fake:.2f}, {np.mean(preds_fake)}\n"
            )

        if self.opt.tracemalloc:
            snapshot = tracemalloc.take_snapshot()
            print("Tracemalloc: validation memory")
            util.print_snapshot(snapshot)
