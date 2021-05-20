import torch
from .base_model import BaseModel
from . import networks
from torchvision.models import resnet18
from util import util


class ClassifierModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='move_eval', netG='classifier', data_root="datasets/ROOM_composite", preprocess="resize", load_size=64, crop_size=64, name="Classifier", no_flip=True)  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        # parser.add_argument('--use_resnet18', action="store_true", help='If specified, use a Resnet18 model instead of own classification')
        # parser.add_argument('--use_pretrained', action="store_true", help='If specified, use a pretrained version of Resnet. Only possible in combination with --use_resnet18')

        parser.add_argument('--model_type', default="default", help='Type of classifier model used: choose from default, Resnet18, or Resnet18_pretrained', choices=["default", "Resnet18", "Resnet18_pretrained"])

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
        self.loss_names = ['loss_real', 'loss_move', 'loss_random', 'loss_scanline', 'loss', "acc_real", "acc_move", "acc_scanline", "acc_random"]

        for l in self.loss_names:
            setattr(self, l, 0)
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real', 'move', 'random', 'scanline']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Classifier']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.

        # create confusion matrix
        self.reset_conf_matrix()


        self.netClassifier = networks.define_D(opt.input_nc, opt.ngf, "classifier", gpu_ids=self.gpu_ids, num_classes=4, classifier_type=opt.model_type)

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            self.CELoss = torch.nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(self.netClassifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # define the four different categories

        self.real = input['real']
        self.move = input['move']
        self.random = input['random']
        self.scanline = input['scanline']


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        self.pred_real = self.netClassifier(self.real)
        self.pred_move = self.netClassifier(self.move)
        self.pred_random = self.netClassifier(self.random)
        self.pred_scanline = self.netClassifier(self.scanline)


    def get_accuracies(self):
        acc = self.confusion_matrix.diagonal().sum() / self.confusion_matrix.sum()
        self.acc_real, self.acc_move, self.acc_random, self.acc_scanline = self.confusion_matrix.diag()/self.confusion_matrix.sum(1)
        return acc

    def update_conf_matrix(self):
        for class_, pred in enumerate([self.pred_real, self.pred_move, self.pred_random, self.pred_scanline]):
                argmax_pred = pred.max(dim=1)[1]
                for p in argmax_pred:
                    self.confusion_matrix[class_, p] += 1

    def reset_conf_matrix(self):
        self.confusion_matrix = torch.zeros(4, 4)


    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        B = self.pred_move.shape[0]
        y_base = torch.ones(B).long().to(self.device)

        self.loss_real = self.CELoss(self.pred_real, y_base * 0)
        self.loss_move = self.CELoss(self.pred_move, y_base * 1)
        self.loss_random = self.CELoss(self.pred_random, y_base * 2)
        self.loss_scanline = self.CELoss(self.pred_scanline, y_base * 3)

        self.loss = self.loss_real + self.loss_move + self.loss_random + self.loss_scanline

        self.loss.backward()

        # TODO: save the losses and the predictions

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results

        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G


    def run_batch(self, data, overall_batch):
        self.reset_conf_matrix()
        self.set_input(data)
        self.optimize_parameters()
        self.update_conf_matrix()
        acc = self.get_accuracies()

        if overall_batch % self.opt.print_freq == 0:
            print("accuracy:", acc)
            print(self.confusion_matrix)


    def test(self, data):
        assert not self.isTrain, "Model should be in testing state"

        with torch.no_grad():
            self.set_input(data)

            self.forward()
            # update confusion matrix
            self.update_conf_matrix()

    def display_test(self, batch, webpage):
        pass

    def print_results(self, total_nr_batches):
        print(f"Confusion matrix:\n{self.confusion_matrix}")
        print(f"Overall accuracy: {self.get_accuracies():.2f}")
        print(f"statistics per class: real, move, random, scanline: \
            \nAccuracy: {self.confusion_matrix.diag()/self.confusion_matrix.sum(1)} \
            \nPredicted per class: {self.confusion_matrix.sum(0)}\
            \nGT # instances per class: {self.confusion_matrix.sum(1)} ")

        # save plot
        util.plot_confusion_matrix(self.confusion_matrix, self.visual_names, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'confusion_matrix.png'))


