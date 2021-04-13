from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen (in batches)')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=100, help='frequency of saving training results to html (in batches')
        parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console (in batches)')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters

        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.', choices = ["vanilla", "lsgan" , "wgangp"])
        parser.add_argument("--real_target", type=float, default=0.8, help="Target label for the discriminator, can be set <1 to prevent overfitting")
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument("--val_freq", type=int, default=100, help="every val_freq batches run the model on validation data, and obtain accuracies for training schedule.",)
        parser.add_argument("--val_batch_size", type=int, default=128, help="every val_freq batches run the model on validation data, and obtain accuracies for training schedule")
        parser.add_argument("--tracemalloc", action="store_true", help="If specified, largest memory allocations are printed")
        parser.add_argument('--min_obj_surface', type=int, default=100, help= "Minimum number of pixels an object needs to be to be eligible for moving")
        parser.add_argument("--use_amp", action="store_true", help="If specified, gradient scaling using AMP GradScaler is enabled")
        parser.add_argument("--noisy_labels", action="store_true", help="If specified, random noise will be added to the target labels in the adversarial loss",)
        parser.add_argument("--fake_target",type=float, default=0.1, help="Soft labeling for fake targets")

        self.isTrain = True
        return parser
