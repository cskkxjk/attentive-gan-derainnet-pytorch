'''
All net structure related files are saved in this folder!
'''

from .generator import *
from .discriminator import *
from .vgg_init import *

class BlendModel(nn.Module):
    '''Blend model class
    '''

    def __init__(self, device=None, generator=None, discriminator=None, **kwargs):
        super(BlendModel, self).__init__()
        if discriminator is not None:
            self.discriminator = discriminator.to(device)
        else:
            self.discriminator = None
        if generator is not None:
            self.generator = generator.to(device)
        else:
            self.generator = None

    def forward(self, batch_size, **kwargs):
        raise NotImplementedError

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model