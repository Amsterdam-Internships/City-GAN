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
        parser.set_defaults(dataset_mode='room', preprocess="resize_and_crop", load_size=64, crop_size=64, no_flip=True, )  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

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

        self.loss_names = []

        # define variables for plotting and saving
        self.visual_names = ["tgt", "obj", "composite"]

        # define the convnet that predicts theta
        self.ConvNet = networks.MoveConvNET()

        self.model_names = ["ConvNet"]

        if self.isTrain:  # only defined during training time
            self.netD = networks.define_D(opt.input_nc, ndf, opt.netD)
            self.model_names.append("netD")

            self.criterionGAN = networks.GANLoss("vanilla")


        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks







    def set_input(self, input):
        """
        What do we need:

            - centered object + object mask (from source)
            - some real image (e.g. source)
            - target image

        """
        self.img = input['img']
        self.nr_masks = int(input["nr_masks"])

        self.tgt = None
        self.obj = None
        self.src = None
        self.obj_mask = None


        # assign the masks to the model
        for i in range(self.nr_masks):
            setattr(self, f"mask_{i}", input[f"mask{i}"])


    def forward(self):
        """
        what needs to be done:
            - target image and centered object are fed to convnet (initialized in the init)
            - theta parameters are the output
            - affine transformation on the object and the object mask
            - the transformed object and object masks are composited --> output img
        """
        # concatenate the target and object on channel dimension
        tgt_obj_concat = torch.cat(self.tgt, self.obj, 1)

        # compute theta using the convolutional network
        self.theta = self.ConvNet(tgt_obj_concat)

        # make sure theta is scaled: preventing object from moving outside img

        # use theta to transform the object and the mask
        # is affine the correct function? perhaps we should use affine_grid
        self.transf_obj = affine(self.obj, angle=, scale=, shear=)
        self.transf_obj_mask = affine(self.obj_mask, angle=, scale=, shear=)


        # composite the moved object with the background from the target
        self.composite, _ = networks.composite_image(self.transf_obj, self.tgt, self.transf_obj_mask)

        # get the prediction on the fake image
        self.pred_fake = self.netD(self.composite)

        # get the prediction on the real image
        self.pred_real = self.netD(self.src)




    def backward(self):
        """
        todo here
            - compute the losses (GAN loss real fake and possibly more)
            - backward over the loss for D
            - backward over the loss to update the ConvNet
            - perhaps use AMP scaler again?

        """

        pass


    def optimize_parameters(self, data, overall_batch):
        """Update network weights; it will be called in every training iteration.
        """
        # run the forward pass
        self.forward()


        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G







    def baseline(self, data, obj_idx = -1, type_='random'):

        assert type_ in {"random", "scanline"}, f"Type {type_} not recognized, choose from \"random\" or \"scanline\""

        self.set_input(data)

        # edge case, what if the user sets the obdj_idx in [0, 4]?
        if obj_idx == -1:
            # set the minimum mask to 4, as first floor are walls, floor, sky
            obj_idx = np.random.randint(4, self.nr_masks)
            # obj_idx = 9
        print("object idx", obj_idx)

        img_width, img_height = self.img.shape[2:4]

        # obj_mask = self.masks[:, obj_idx]
        obj_mask = getattr(self, f"mask_{obj_idx}")
        obj_mask_binary = (obj_mask > 0).int()
        obj_width = int(torch.max(torch.sum(obj_mask>0, axis=2)))
        obj_height = int(torch.max(torch.sum(obj_mask>0, axis=3)))

        obj_size_approx = obj_width * obj_height

        # divide the image into segments
        # background includes the object to be moved
        background = (1-obj_mask_binary) * self.img
        obj = obj_mask_binary * self.img


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
        self.moved = new_background  * self.img + moved_obj

        print(x_translation, y_translation)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(util.tensor2im(self.img), origin="upper")
        ax1.set_title("original")
        ax2.imshow(util.tensor2im(obj), origin="upper")
        ax2.set_title("object")
        ax3.imshow(util.tensor2im(moved_obj), origin="upper")
        ax3.set_title(f"moved_obj ({x_translation}, {y_translation})")
        ax4.imshow(util.tensor2im(self.moved), origin="upper")
        ax4.set_title("result")
        plt.tight_layout()
        plt.show()


        return self.img, self.moved
