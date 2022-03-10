"""
Custom Trainer Class
"""
import os
import math
import torch.nn
import numpy as np
import cv2.cv2 as cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils.common import BaseTrainer
import logging
from utils.common import toggle_grad, compute_bce, img2mse, mse2psnr
import utils.pytorch_ssim as pytorch_ssim
logger_py = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    ''' Trainer object for a3f.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, vgg, optimizer_g, optimizer_d, device=None, vis_dir=None, overwrite_visualization=True, **kwargs):
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.overwrite_visualization = overwrite_visualization

        self.generator = self.model.generator
        self.discriminator = self.model.discriminator
        self.vgg = vgg
        self.mse = torch.nn.MSELoss()
        self.ssim = pytorch_ssim.SSIM(size_average=False)

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        ''' Performs a training step.
        Args:
            data (tuple): data tuple
            it (int): training iteration
        '''
        loss_g, loss_g_adv, loss_att, loss_m, loss_p = self.train_step_generator(data, it)
        loss_d, loss_d_adv, loss_map = self.train_step_discriminator(data, it)
        return {
            'loss_g': loss_g,
            'loss_g_adv': loss_g_adv,
            'loss_att': loss_att,
            'loss_m': loss_m,
            'loss_p': loss_p,
            'loss_d': loss_d,
            'loss_d_adv': loss_d_adv,
            'loss_map': loss_map,
        }

    def train_step_generator(self, data, it=None):
        rain_real, clean_real, clean_real_half, clean_real_quarter, binary_mask = \
            data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device)
        generator = self.generator
        discriminator = self.discriminator
        vgg = self.vgg
        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        toggle_grad(vgg, False)
        generator.train()
        discriminator.train()
        vgg.eval()
        self.optimizer_g.zero_grad()
        result_g = generator(rain_real)  # mask_list, frame1, frame2, clean_fake
        result_d_fake = discriminator(result_g[3])
        loss_g_adv = compute_bce(result_d_fake[1], 1)
        # loss_g_adv = torch.log(torch.ones_like(result_d_fake[1], device=self.device) - result_d_fake[1]).mean()
        loss_att = 0
        for i in range(len(result_g[0])):
            pow = torch.tensor(math.pow(self.generator.theta, (len(result_g[0]) - 1 - i)), device=self.device)
            loss_att += pow * self.mse(binary_mask, result_g[0][i])

        loss_m = 0
        _s = [result_g[1], result_g[2], result_g[3]]
        _t = [clean_real_quarter, clean_real_half, clean_real]
        for i, ae_lambda in enumerate(self.generator.ae_lambda):
            loss_m += ae_lambda * self.mse(_s[i], _t[i])

        loss_p = 0
        vgg_fake = vgg(result_g[3])
        vgg_real = vgg(clean_real)
        for i in range(len(vgg_fake)):
            loss_p += self.mse(vgg_fake[i], vgg_real[i])

        # loss_g = 0.01 * loss_g_adv + loss_att + loss_m + loss_p
        loss_g = 0.01 * loss_g_adv + loss_att + loss_m + loss_p
        loss_g.backward()
        self.optimizer_g.step()
        return (
            loss_g.item(), loss_g_adv.item(), loss_att.item(), loss_m.item(), loss_p.item()
        )

    def train_step_discriminator(self, data, it=None):
        rain_real, clean_real = data[0].to(self.device), data[1].to(self.device)
        generator = self.generator
        discriminator = self.discriminator
        vgg = self.vgg
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        toggle_grad(vgg, False)
        vgg.eval()
        generator.train()
        discriminator.train()
        self.optimizer_d.zero_grad()
        result_g = generator(rain_real)  # mask_list, frame1, frame2, clean_fake
        result_d_fake = discriminator(result_g[3])
        result_d_real = discriminator(clean_real)
        loss_d_adv = compute_bce(result_d_fake[1], 0) + compute_bce(result_d_real[1], 1)
        # loss_d_adv = -torch.log(result_d_real[1]) - torch.log(torch.ones_like(result_d_fake[1], device=self.device) - result_d_fake[1])
        loss_map = self.mse(result_d_fake[0], result_g[0][3]) + self.mse(result_d_real[0], torch.zeros_like(result_d_real[0], device=self.device))
        loss_d = loss_d_adv + 0.05 * loss_map
        loss_d.backward()
        self.optimizer_d.step()
        return (
            loss_d.item(), loss_d_adv.item(), loss_map.item()
        )

    def evaluate(self, dataloader):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        generator = self.generator
        generator.eval()
        ssim_list = []
        psnr_list = []
        with torch.no_grad():
            for batch in dataloader:
                rain_real, clean_real = batch[0].to(self.device), batch[1].to(self.device)
                mask_list, frame1, frame2, clean_fake = generator(rain_real)
                ssim = self.ssim(clean_real, clean_fake)
                psnr = mse2psnr(img2mse(clean_real, clean_fake))
                ssim_list.append(ssim)
                psnr_list.append(psnr)
            ssim_mean = torch.cat(ssim_list, dim=0).mean().item()
            psnr_mean = torch.cat(psnr_list, dim=0).mean().item()
        eval_dict = {
            'ssim': ssim_mean,
            'psnr': psnr_mean
        }
        return eval_dict

    def visualize(self, data, it=0):
        ''' Visualized the data.
        Args:
            it (int): training iteration
        '''
        rain_real, clean_real = data[0].to(self.device), data[1].to(self.device)
        generator = self.generator
        generator.eval()
        with torch.no_grad():
            mask_list, frame1, frame2, clean_fake = generator(rain_real)  # mask_list, frame1, frame2, clean_fake

        if self.overwrite_visualization:
            out_file_name = 'visualization.jpg'
        else:
            out_file_name = 'visualization_%010d.jpg' % it
        rain_real = rain_real.detach().cpu()
        clean_real = clean_real.detach().cpu()
        attention_map_0 = mask_list[0][0].detach().cpu()
        attention_map_1 = mask_list[1][0].detach().cpu()
        attention_map_2 = mask_list[2][0].detach().cpu()
        attention_map_3 = mask_list[3][0].detach().cpu()
        clean_fake = clean_fake.detach().cpu()
        temp_map_0 = torch.zeros_like(attention_map_0)
        temp_map_1 = torch.zeros_like(attention_map_0)
        colored_att_map_0 = torch.cat([temp_map_0, temp_map_1, attention_map_0], dim=0)
        colored_att_map_1 = torch.cat([temp_map_0, temp_map_1, attention_map_1], dim=0)
        colored_att_map_2 = torch.cat([temp_map_0, temp_map_1, attention_map_2], dim=0)
        colored_att_map_3 = torch.cat([temp_map_0, temp_map_1, attention_map_3], dim=0)
        image_list = [rain_real[0], clean_real[0], colored_att_map_0, colored_att_map_1, colored_att_map_2, colored_att_map_3, clean_fake[0]]
        image_concat = torch.cat(image_list, dim=2)
        save_image(image_concat.data.cpu(), os.path.join(self.vis_dir, out_file_name), nrow=1, padding=0)
        return image_concat.data.cpu()
