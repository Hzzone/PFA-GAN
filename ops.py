from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from apex.parallel import DistributedDataParallel
from typing import Union
from apex import amp
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        m.weight.data.normal_(0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.zero_()


def to_ddp(modules: Union[list, nn.Module], optimizer: torch.optim.Optimizer = None, opt_level: int = 0) -> Union[
    DistributedDataParallel, tuple]:
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    if optimizer is not None:
        modules, optimizer = amp.initialize(modules, optimizer, opt_level="O{}".format(opt_level), verbosity=1)
    if isinstance(modules, list):
        modules = [DistributedDataParallel(x, delay_allreduce=True) for x in modules]
    else:
        modules = DistributedDataParallel(modules, delay_allreduce=True)
    if optimizer is not None:
        return modules, optimizer
    else:
        return modules


def age2group(age, age_group):
    if isinstance(age, np.ndarray):
        groups = np.zeros_like(age)
    else:
        groups = torch.zeros_like(age).to(age.device)
    if age_group == 4:
        section = [30, 40, 50]
    elif age_group == 5:
        section = [20, 30, 40, 50]
    elif age_group == 7:
        section = [10, 20, 30, 40, 50, 60]
    else:
        raise NotImplementedError
    for i, thresh in enumerate(section, 1):
        groups[age > thresh] = i
    return groups


def get_norm_layer(norm_layer, module, **kwargs):
    if norm_layer == 'none':
        return module
    elif norm_layer == 'bn':
        return nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'in':
        return nn.Sequential(
            module,
            nn.InstanceNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'sn':
        return nn.utils.spectral_norm(module, **kwargs)
    else:
        return NotImplementedError


def group2onehot(groups, age_group):
    code = torch.eye(age_group)[groups.squeeze()]
    if len(code.size()) > 1:
        return code
    return code.unsqueeze(0)


def group2feature(group, age_group, feature_size):
    onehot = group2onehot(group, age_group)
    return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)


def get_dex_age(pred):
    pred = F.softmax(pred, dim=1)
    value = torch.sum(pred * torch.arange(pred.size(1)).to(pred.device), dim=1)
    return value


def pfa_encoding(source, target, age_group):
    source, target = source.long(), target.long()
    code = torch.zeros((source.size(0), age_group - 1, 1, 1, 1)).to(source)
    for i in range(source.size(0)):
        code[i, source[i]: target[i], ...] = 1
    return code


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root):
        assert dist.is_initialized()
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        for i in range(len(stats)):
            var = stats[i]
            if isinstance(var, dict):
                group_name = list(var.keys())[0]
                for j in range(len(var[group_name])):
                    scalar_name = get_varname(var[group_name][j])
                    scalar = reduce_tensor(var[group_name][j].detach().mean(), self.world_size).item()
                    output_str += '{} {:2.5f}, '.format(scalar_name, scalar)
                    self.writer.add_scalars(group_name, {
                        scalar_name: scalar
                    }, step)
            else:
                var_name = get_varname(stats[i])
                var = reduce_tensor(var.detach().mean(), self.world_size).item()
                output_str += '{} {:2.5f}, '.format(var_name, var)
        if self.local_rank == 0:
            print(output_str)

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.jpg'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim_loss(img1, img2, window_size=11):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def normalize(input, mean, std):
    mean = torch.Tensor(mean).to(input.device)
    std = torch.Tensor(std).to(input.device)
    return input.sub(mean[None, :, None, None]).div(std[None, :, None, None])
