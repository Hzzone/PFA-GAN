import torch.utils.data as tordata
import torchvision
import os.path as osp
from networks import PatchDiscriminator, AuxiliaryAgeClassifier, Generator
import torch.nn.functional as F
import torch
from ops import get_dex_age
from ops import age2group, weights_init, ls_gan, load_network
from ops import to_ddp, compute_ssim_loss, normalize, LoggerX
import numpy as np
from dataset import GroupDataset, PFADataset, data_prefetcher
from apex import amp


class PFA_GAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.prefetcher = self.get_train_loader()
        self.test_images = self.get_test_images()
        self.logger = LoggerX(osp.join('materials', 'checkpoints', opt.name))
        self.init_model()

    def get_test_images(self):
        opt = self.opt
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        test_dataset = GroupDataset(
            age_group=opt.age_group,
            train=False,
            group=opt.source,
            dataset_name=opt.dataset_name,
            transforms=transforms)
        test_sampler = tordata.distributed.DistributedSampler(test_dataset, shuffle=False)
        test_loader = tordata.DataLoader(
            dataset=test_dataset,
            batch_size=opt.batch_size * 5,
            shuffle=False,
            drop_last=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=test_sampler
        )
        return next(iter(test_loader)).cuda()

    def get_train_loader(self):
        opt = self.opt
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.pretrained_image_size),
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = PFADataset(
            age_group=opt.age_group,
            max_iter=opt.max_iter,
            batch_size=opt.batch_size * len(opt.device_ids),
            dataset_name=opt.dataset_name,
            source=opt.source,
            transforms=transforms)
        train_sampler = tordata.distributed.DistributedSampler(train_dataset, shuffle=False)

        train_loader = tordata.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            drop_last=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        # source_img, true_img, source_label, target_label, true_label, true_age, mean_age
        return data_prefetcher(train_loader, [0, 1])

    def init_model(self):
        opt = self.opt
        generator = Generator(norm_layer='in', age_group=opt.age_group)
        generator.apply(weights_init)
        discriminator = PatchDiscriminator(age_group=opt.age_group,
                                           repeat_num=int(np.log2(opt.image_size) - 4), norm_layer='sn')

        vgg_face = torchvision.models.vgg16(num_classes=2622)
        vgg_face.load_state_dict(load_network(osp.join('materials', 'DeepFaceVGG_RGB.pth')))
        vgg_face = vgg_face.features[:23]
        vgg_face.eval()

        age_classifier = AuxiliaryAgeClassifier(age_group=opt.age_group,
                                                repeat_num=int(np.log2(opt.pretrained_image_size) - 4))
        age_classifier.load_state_dict(load_network(osp.join('materials',
                                                             'dex_simple_{}_{}_age_classifier.pth'.format(
                                                                 opt.dataset_name, opt.pretrained_image_size))))
        age_classifier.eval()

        d_optim = torch.optim.Adam(discriminator.parameters(), opt.init_lr, betas=(0.5, 0.99))
        g_optim = torch.optim.Adam(generator.parameters(), opt.init_lr, betas=(0.5, 0.99))

        self.logger.modules = [generator, discriminator, d_optim, g_optim]

        if opt.restore_iter > 0:
            self.logger.load_checkpoints(opt.restore_iter)

        self.generator, self.g_optim = to_ddp(generator, g_optim)
        self.discriminator, self.d_optim = to_ddp(discriminator, d_optim)
        self.vgg_face = to_ddp(vgg_face)
        self.age_classifier = to_ddp(age_classifier)

    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.generator.eval()
        real_img = self.test_images
        bs, ch, w, h = real_img.size()
        fake_imgs = [real_img, ]
        # generate fake images
        for target in range(opt.source + 1, opt.age_group):
            output = self.generator(real_img, torch.ones(bs) * opt.source, torch.ones(bs) * target)
            fake_imgs.append(output)
        fake_imgs = torch.stack(fake_imgs).transpose(1, 0).reshape((-1, ch, w, h))

        fake_imgs = fake_imgs * 0.5 + 0.5
        grid_img = torchvision.utils.make_grid(fake_imgs.clamp(0., 1.), nrow=opt.age_group - opt.source)
        self.logger.save_image(grid_img, n_iter, 'test')

    def fit(self):
        opt = self.opt
        for n_iter in range(opt.restore_iter + 1, opt.max_iter + 1):
            inputs = self.prefetcher.next()
            self.train(inputs, n_iter)
            if n_iter % opt.save_iter == 0 or n_iter == opt.max_iter:
                # self.logger.checkpoints(n_iter)
                self.generate_images(n_iter)

    def age_criterion(self, input, gt_age):
        opt = self.opt
        age_logit, group_logit = self.age_classifier(input)
        return F.mse_loss(get_dex_age(age_logit), gt_age) + \
               F.cross_entropy(group_logit, age2group(gt_age, opt.age_group).long())

    def extract_vgg_face(self, inputs):
        inputs = normalize((F.hardtanh(inputs) * 0.5 + 0.5) * 255,
                           [129.1863, 104.7624, 93.5940],
                           [1.0, 1.0, 1.0])
        return self.vgg_face(inputs)

    def train(self, inputs, n_iter):
        opt = self.opt
        source_img, true_img, source_label, target_label, true_label, true_age, mean_age = inputs
        self.generator.train()
        self.discriminator.train()

        if opt.image_size < opt.pretrained_image_size:
            source_img_small = F.interpolate(source_img, opt.image_size)
            true_img_small = F.interpolate(true_img, opt.image_size)
        else:
            source_img_small = source_img
            true_img_small = true_img
        g_source = self.generator(source_img_small, source_label, target_label)
        if opt.image_size < opt.pretrained_image_size:
            g_source_pretrained = F.interpolate(g_source, opt.pretrained_image_size)
        else:
            g_source_pretrained = g_source

        ########Train D###########
        self.d_optim.zero_grad()
        d1_logit = self.discriminator(true_img_small, true_label)
        # d2_logit = self.discriminator(true_img, source_label)
        d3_logit = self.discriminator(g_source.detach(), target_label)

        # d_loss = 0.5 * (ls_gan(d1_logit, 1.) + ls_gan(d2_logit, 0.) + ls_gan(d3_logit, 0.))
        d_loss = 0.5 * (ls_gan(d1_logit, 1.) + ls_gan(d3_logit, 0.))

        with amp.scale_loss(d_loss, self.d_optim) as scaled_loss:
            scaled_loss.backward()
        self.d_optim.step()

        ########Train G###########
        self.g_optim.zero_grad()
        ################################GAN_LOSS##############################
        gan_logit = self.discriminator(g_source, target_label)
        # g_loss = 0.5 * ls_gan(gan_logit, 1.)
        g_loss = ls_gan(gan_logit, 1.)

        ################################Age_Loss##############################
        age_loss = self.age_criterion(g_source_pretrained, mean_age)

        ################################L1_loss##############################
        l1_loss = F.l1_loss(g_source_pretrained, source_img)

        ################################SSIM_loss##############################
        ssim_loss = compute_ssim_loss(g_source_pretrained, source_img, window_size=10)

        ################################ID_loss##############################
        id_loss = F.mse_loss(self.extract_vgg_face(g_source_pretrained), self.extract_vgg_face(source_img))

        pix_loss_weight = max(opt.pix_loss_weight,
                              opt.pix_loss_weight * (opt.decay_pix_factor ** (n_iter // opt.decay_pix_n)))

        total_loss = g_loss * opt.gan_loss_weight + \
                     (l1_loss * (1 - opt.alpha) + ssim_loss * opt.alpha) * pix_loss_weight + \
                     id_loss * opt.id_loss_weight + \
                     age_loss * opt.age_loss_weight

        with amp.scale_loss(total_loss, self.g_optim) as scaled_loss:
            scaled_loss.backward()
        self.g_optim.step()

        self.logger.msg([
            d1_logit, d3_logit, gan_logit, age_loss, l1_loss, ssim_loss, id_loss
        ], n_iter)
