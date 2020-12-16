### PFA-GAN: Progressive Face Aging with Generative Adversarial Network

[arxiv](https://arxiv.org/abs/2012.03459)

This paper has been accepted as a Regular Paper on **IEEE Transactions on Information Forensics & Security**, to be published soon later.

#### Requirements

* PyTorch 1.51+ (Ubuntu, cuda 10.2)
* [Nvidia Apex](https://github.com/NVIDIA/apex) (install from source)

#### Training

1. Downloading the datasets.
2. Aligning and generating data using the ages estimated by Face++ APIs.
3. Training with four GPUs on single node:

```shell
python -m torch.distributed.launch --nproc_per_node=4 --master_port=17674 \\
      main.py --device_ids 0 1 2 3 --source 0
```
4. The pre-trained model is to come.

### Citation

If you found this code or our work useful please cite us:
```
@misc{huang2020pfagan,
      title={PFA-GAN: Progressive Face Aging with Generative Adversarial Network}, 
      author={Zhizhong Huang and Shouzhen Chen and Junping Zhang and Hongming Shan},
      year={2020},
      eprint={2012.03459},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
