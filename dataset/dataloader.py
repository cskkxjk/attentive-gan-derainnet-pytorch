import os
from torch.utils import data
from dataset.dataset import RainDropDataset

def get_loader(basename):
    # if basename.find('absorption') != -1:
    #     Data_Reader = AbsorptionTrainDataset
    # elif basename.find('synthetic') != -1:
    #     Data_Reader = SynthesisDataset
    # elif basename.find('DSLR') != -1:
    #     Data_Reader = UnalignedDataset
    # elif basename.find('SIR') != -1 or basename.find('Postcard') != -1 or basename.find(
    #         'SolidObject') != -1 or basename.find('Wild') != -1:
    #     Data_Reader = SIR2Dataset
    # elif basename.find('ZN18') != -1:
    #     Data_Reader = ZN18Dataset
    # elif basename.find('LY20') != -1:
    #     Data_Reader = LY20Dataset
    # else:
    #     raise NotImplementedError

    return RainDropDataset

def get_loader_train(config):
    if config.train_dir is None:
        return None
    basename = os.path.basename(config.train_dir)
    data_reader = get_loader(basename)
    dataset = data_reader(config.train_dir, img_size=128, length=config.length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader

def get_loader_val(config):
    if config.val_dir is None:
        return None
    basename = os.path.basename(config.val_dir)
    data_reader = get_loader(basename)
    dataset = data_reader(config.val_dir, img_size=128, length=config.length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader


def get_loader_test(config):
    if config.test_dir is None:
        return None
    basename = os.path.basename(config.test_dir)
    data_reader = get_loader(basename)
    dataset = data_reader(config.test_dir, img_size=128, length=config.length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader
