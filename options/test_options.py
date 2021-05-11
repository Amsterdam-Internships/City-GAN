from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')
        parser.add_argument('--display_freq', type=int, default=100, help='how many test images to run')
        parser.add_argument('--min_iou', type=float, default=0.5, help='Minimum IOU for a discovery to be successful')
        # for generating data for MoveGAN classifier
        parser.add_argument('--data_phase', type=str, default="train", help='Phase to save the generated data to, can be test or train')

        self.isTrain = False
        return parser
