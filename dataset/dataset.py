import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def make_power(img, base=4):
    # Make sure that the image's size is scaled to base's power.
    w, h = img.size
    new_w = (w // base) * base
    new_h = (h // base) * base
    resize_operation = transforms.Resize([new_h, new_w])
    img = resize_operation(img)
    return img

def get_binary_mask(img, back_gt):
    """
    Get binary mask
    :param img: raindrop image
    :param back_gt: clean image
    :return:
    """
    _mean_image = np.mean(img, 2, dtype=np.float32)
    _mean_back_gt = np.mean(back_gt, 2, dtype=np.float32)
    _diff = np.abs(_mean_image - _mean_back_gt)
    _diff[_diff <= 28] = 0
    _diff[_diff > 28] = 1
    # torch.from_numpy(_diff zeng)
    _diff = _diff[:, :, np.newaxis]
    return _diff

class RainDropDataset(data.Dataset):
    def __init__(self, root_path, img_size=128, length=None):
        self.root_path = root_path
        self.rain_path = os.path.join(root_path, 'data')
        self.clean_path = os.path.join(root_path, 'gt')
        self.resize = transforms.Resize([240, 360])
        self.half_resize = transforms.Resize([120, 180])
        self.quarter_resize = transforms.Resize([60, 90])
        self.to_tensor = transforms.ToTensor()
        self.i_files = []
        self.b_files = []
        files = os.listdir(self.rain_path)
        for fname in files:
            prefix = fname.split('_')[0]
            suffix = fname.split('.')[1]
            if suffix == 'png' or suffix == 'jpg':
                i_path = os.path.join(self.rain_path, fname)
                b_path = os.path.join(self.clean_path, prefix + '_clean.' + suffix)
                self.i_files.append(i_path)
                self.b_files.append(b_path)

        assert len(self.i_files) == len(self.b_files), 'I and B Size not equal!'
        if length is not None:
            self.i_files = self.i_files[:length]
            self.b_files = self.b_files[:length]
        self.length = len(self.i_files)

    def __getitem__(self, index):
        image_clean = Image.open(self.b_files[index]).convert('RGB')
        image_rain = Image.open(self.i_files[index]).convert('RGB')
        image_clean = make_power(image_clean, base=4)
        image_rain = make_power(image_rain, base=4)
        img_clean = self.resize(image_clean)
        img_clean_half = self.half_resize(image_clean)
        img_clean_quarter = self.quarter_resize(image_clean)
        img_rain = self.resize(image_rain)
        binary_mask = get_binary_mask(img_rain, img_clean)

        return self.to_tensor(img_rain), self.to_tensor(img_clean), self.to_tensor(img_clean_half), self.to_tensor(img_clean_quarter), self.to_tensor(binary_mask)

    def __len__(self):
        return self.length
