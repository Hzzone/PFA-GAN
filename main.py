import argparse
import torch
import torch.distributed as dist
from models import PFA_GAN

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', '-d', nargs='*', help='devices id to be used', type=int,
                    default=list(range(torch.cuda.device_count())))
parser.add_argument("--age_group", help='divide age group', default=4, type=int)
parser.add_argument("--image_size", help='input image size', default=256, type=int)
parser.add_argument("--pretrained_image_size", help='input image size', default=256, type=int)
parser.add_argument("--source", help='source age group, only works for GANs-based methods or PFA-GAN', type=int,
                    default=0)
parser.add_argument("--num_workers", help='num_workers', default=16, type=int)
parser.add_argument("--restore_iter", help='restore_iter', default=0, type=int)
parser.add_argument("--local_rank", help='local process rank, not need to be set.', default=0, type=int)
parser.add_argument("--dataset_name", default='morph', type=str)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--max_iter", default=200000, type=int)
parser.add_argument("--save_iter", help='checkpoint iter', default=2000, type=int)
parser.add_argument("--init_lr", default=1e-4, type=float)
parser.add_argument("--gan_loss_weight", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--pix_loss_weight", type=float)
parser.add_argument("--id_loss_weight", type=float)
parser.add_argument("--age_loss_weight", type=float)
parser.add_argument("--decay_pix_factor", default=0, type=float)
parser.add_argument("--decay_pix_n", default=2000, type=int)
parser.add_argument("--name", default='morph', type=str)
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    torch.cuda.set_device(opt.device_ids[opt.local_rank])
    dist.init_process_group(backend='nccl',
                            init_method='env://')

    model = PFA_GAN(opt)
    model.fit()
