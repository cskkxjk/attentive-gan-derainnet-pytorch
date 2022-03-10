import numpy as np
import torch
import os
import random
import logging
import torch.nn.functional as F


def set_device(device_num):
    if device_num == -1 or torch.cuda.is_available() == False:
        device = torch.device('cpu')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:{}'.format(int(device_num)))
    return device

# random seed controller
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = torch.cuda.is_available()

# default logger setter
def set_logger(config):
    logfile = os.path.join(config.out_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_bce(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy(d_out, targets)
    return loss


img2mse = lambda x, y: torch.mean(((x - y) ** 2).reshape(x.shape[0], -1), 1)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

# Base Trainer class
class BaseTrainer(object):
    """ Base trainer class.
    """
    def evaluate(self, *args, **kwargs):
        """ Performs an evaluation.
        """
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        """ Performs a training step.
        """
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs an evaluation step.
        """
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """ Performs  visualization.
        """
        raise NotImplementedError
