import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from math import log


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], img_dim=64):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'copy':
        net = CopyUNet(input_nc, output_nc, norm_layer=norm_layer, dropout=use_dropout, img_dim=img_dim)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[], img_dim=64):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == "copy":
        net = CopyUNet(input_nc, 1, norm_layer=norm_layer, discriminator=True, img_dim=img_dim)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)




###############################################
# HELPER FUNCTIONS
###############################################

def composite_image(src, tgt, mask=None):
    """
    Crop out the mask in the source image and paste on the tgt image
    If no mask is given, A random polygon is generared to generate a
    grounded fake
    """
    def get_polygon_mask(w, h, b=1, coverage=0.1):
        mask = torch.zeros((b, 1, w, h))
        # TODO: change this to real polygon
        for i in range(b):
            xy = (torch.rand(2) * 0.5 * torch.tensor([w, h])).int()
            mask[i, 0, xy[0]:xy[0]+25, xy[1]:xy[1]+25] = 1

        return mask


    assert src.shape == tgt.shape

    if not torch.is_tensor(mask):
        b, w, h = src.shape[0], src.shape[2], src.shape[3]
        mask = get_polygon_mask(w, h, b)


    # make sure the mask size corresponds to the image size
    for i in [0, 2, 3]:
        assert mask.shape[i] == src.shape[i]

    # compute the composite image based on the mask and inverse mask
    inv_mask = 1 - mask
    composite = torch.mul(src, mask) + torch.mul(tgt, inv_mask)

    return composite, mask


def mask_to_binary(mask):
    """
    Convert a mask in [0, 1] to binary ( in {0, 1})
    """

    assert (mask.min().item() >= 0) and (mask.max().item() <= 1)
    bin_mask = F.relu(torch.sign(mask - 0.5))

    return bin_mask



##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    # TODO: set real label to 0.95/0.7 (in the paper) instead of 1 to prevent overconfidence?
    def __init__(self, gan_mode, target_real_label=0.8, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss






def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps

        return gradient_penalty, gradients
    else:
        return 0.0, None




######################################
# COPYGAN UNET ARCHITECTURE
######################################

class CopyUNet(nn.Module):
    """
    Generator architecture that follows the paper from Arandjelovic et al. 2019
    This implements the CopyPaste architecture from the
    implementation details in the paper
    """

    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, dropout=False, border_zeroing=True, discriminator=False, img_dim=64):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            norm_layer      -- normalization layer
            dropout         -- Use dropout or not
            border_zeroing  -- Set borders of mask to 0
            discriminator   -- in Discriminator mode, score and predicted copy
                                mask are returned
            double_size     -- To handle 128x128 images

        The U-net is constructed from encoder and decoder building blocks
        Inspired from https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/u_net.py
        """
        super(CopyUNet, self).__init__()

        self.discriminator = discriminator
        self.border_zeroing = border_zeroing
        self.img_dim = img_dim

        self.downscale = []
        self.upscale = []

        # if the input size differs from default 64, create down- and
        # upscaling layers before the encoder and after the decoder
        if self.img_dim != 64:
            assert log(self.img_dim, 2).is_integer(), "Image size should be power of 2"

            nr_scale_ops = int(log(self.img_dim, 2) - 6)

            for i in range(nr_scale_ops):
                self.downscale.append(EncoderBlock(input_nc, input_nc, stride=2, kernel=3, padding=1))
                self.upscale.append(DecoderBlock(output_nc, output_nc, stride=1, kernel=3, padding=1))

            self.downscale = nn.Sequential(*self.downscale)
            self.upscale = nn.Sequential(*self.upscale)



        self.enc1 = EncoderBlock(input_nc, 64, stride=1)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(512, 128)
        self.dec2 = DecoderBlock(256, 64)
        self.dec1 = DecoderBlock(128, output_nc, last_layer=True)

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(512, 1)
        self.relu = nn.ReLU()



    def forward(self, input):
        """Standard forward, return decoder output and encoder output if in
        discriminator mode"""
        if self.downscale:
            for layer in self.downscale:
                input = layer(input)

        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))

        if self.upscale:
            for layer in self.upscale:
                dec1 = layer(dec1)

        copy_mask = self.sigmoid(dec1)

        # clamp the borders of the copy mask to 0
        if self.border_zeroing and not self.discriminator:
            border_array = torch.ones_like(copy_mask)
            border_array[:, 0, 0, :] = 0
            border_array[:, 0, -1, :] = 0
            border_array[:, 0, :, 0] = 0
            border_array[:, 0, :, -1] = 0
            copy_mask = copy_mask * border_array

        # return the encoder output if in discriminator mode
        if self.discriminator:
            enc_out = self.avg_pool(enc4).squeeze()
            realness_score = self.sigmoid(self.fc(enc_out))
            out = [realness_score, copy_mask]
        else:
            out = copy_mask


        return out



class EncoderBlock(nn.Module):
    """
    UnetEncoder
    """

    def __init__(self, input_nc, output_nc, stride=2, kernel=3, padding=1, norm_layer=nn.InstanceNorm2d, slope=0.2, dropout=False, use_bias=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(EncoderBlock, self).__init__()

        layers = [
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel, stride=stride, padding=padding, bias=use_bias),
            norm_layer(output_nc),
            nn.LeakyReLU(slope, True)
        ]

        if dropout:
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        """Standard forward"""
        # print("encoder block forward, input shape", input.shape)
        return self.model(input)


class DecoderBlock(nn.Module):
    """
    UnetEncoder
    """

    def __init__(self, input_nc, output_nc, stride=1, kernel=3, padding=1, norm_layer=nn.InstanceNorm2d, slope=0.2, dropout=False, use_bias=False, last_layer=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(DecoderBlock, self).__init__()

        layers = []
        # nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel, stride=stride, padding=padding, bias=use_bias),
        if not last_layer:
            layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.Conv2d(input_nc, output_nc, stride=stride, kernel_size=kernel, padding=padding)]
            layers += [norm_layer(output_nc), nn.LeakyReLU(slope, True)]
        else:
            layers.append(nn.Conv2d(input_nc, output_nc, stride=1, kernel_size=kernel, padding=padding))

        if dropout:
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)


    def forward(self, input):
        """Standard forward"""
        # print("Input shape decoder block:", input.shape)
        return self.model(input)




######################################
# COPYGAN LOSS FUNCTIONS
######################################


class ConfidenceLoss(nn.Module):
    """Defines the confidence loss, inspired from https://github.com/FLoosli/CP_GAN/blob/master/CP_GAN_models.py

     Penalizes values that are not close to zero or one.
    """
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def __call__(self, pred_mask):
        """Calculate loss given mask.

        Parameters:
            predicted mask (tensor) - predicted mask by generator
        Returns:
            the calculated loss.
        """

        loss = torch.mean(torch.min(pred_mask, 1-pred_mask))

        return loss


class DistinctMaskLoss(nn.Module):
    """Defines the confidence loss, inspired from https://github.com/FLoosli/CP_GAN/blob/master/CP_GAN_models.py

     Penalizes values that are not close to zero or one.
    """
    def __init__(self, nr_obj_classes=3):
        super(DistinctMaskLoss, self).__init__()

        self.nr_obj_classes = nr_obj_classes
        # self.criterion = nn.MSELoss()

    def __call__(self, pred_mask):
        """
        Calculate the overlap between the mask channels
        """
        assert pred_mask.shape[1] == self.nr_obj_classes, "There are not \
            enough masks for all object classes"

        summed_mask = torch.sum(pred_mask, dim=1)

        # get total number of elements in mask and compute fraction over 1
        size = summed_mask.numel()
        fraction_over_one = summed_mask[summed_mask > 1].numel() / size

        return fraction_over_one




class MaskLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """
        Initialize MaskLoss class
        """
        super(MaskLoss, self).__init__()

        self.MSE_loss = nn.MSELoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss()

    def get_mask_loss(self, pred_mask, mask):

        # TODO: should be cross entropy loss, but unofficial github uses MSE
        # compute average pixel-wise loss between two images
        loss = torch.min(self.MSE_loss(pred_mask, mask), self.MSE_loss(pred_mask, (1-mask)))


        return loss


    def __call__(self, mask_real, mask_fake, mask_anti_sc, mask_gr_fake, g_mask, mask_gf):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            - predicted masks by the discriminator for every input image
            - real generator masks on normal input and grounded fake

        Returns:
            the calculated auxilliary loss.
        """

        L_real = self.get_mask_loss(mask_real, torch.zeros_like(mask_real))
        L_fake = self.get_mask_loss(mask_fake, g_mask)
        L_anti_sc = self.get_mask_loss(mask_anti_sc, g_mask)
        L_gr_fake = self.get_mask_loss(mask_gr_fake, mask_gf)

        total_loss = L_real + L_fake + L_anti_sc + L_gr_fake

        return total_loss



######################################
# OTHER ARCHITECTURES
######################################


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


