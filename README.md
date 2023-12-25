# Unoffical Consistency Models for Pytorch (CIFAR-10)

This is the codebase for [Consistency Models](https://arxiv.org/abs/2303.01469), implemented using Pytorch for our experiments on CIFAR-10. We have modified the code to streamline diffusion model training, with additional implementations for consistency distillation, consistency training, and various sampling & editing algorithms included in the paper.

For code and checkpoints for experiments on ImageNet-64, LSUN Bedroom-256, and LSUN Cat-256, check [openai/consistency_models](https://github.com/openai/consistency_models).

The repository for CIFAR-10 experiments is in JAX and can be found at [openai/consistency_models_cifar10](https://github.com/openai/consistency_models_cifar10).

# Pre-trained models

Here are the download links for each model checkpoint:
 * EDM on CIFAR-10: [edm_cifar10_ema]
 * CD on CIFAR-10 with l1 metric: [cd-l1]
 * CD on CIFAR-10 with l2 metric: [cd-l2]
 * CD on CIFAR-10 with LPIPS metric: [cd-lpips]
 * CT on CIFAR-10 with adaptive schedules and LPIPS metric: [ct-lpips]
 * Continuous-time CD on CIFAR-10 with l2 metric: [cifar10-continuous-cd-l2]
 * Continuous-time CD on CIFAR-10 with l2 metric and stopgrad: [cifar10-continuous-cd-l2-stopgrad]
 * Continuous-time CD on CIFAR-10 with LPIPS metric and stopgrad: [cifar10-continuous-cd-lpips-stopgrad]
 * Continuous-time CT on CIFAR-10 with l2 metric: [continuous-ct-l2]
 * Continuous-time CT on CIFAR-10 with LPIPS metric: [continuous-ct-lpips]

OneDrive links:https://1drv.ms/f/s!Avmh265yECFLbRR7rCWMnAiZDfA?e=UuJyn7
Google Drive links:https://drive.google.com/drive/folders/1R8_G8jdiJfQSYfB8VTozGSvmMxGHGQOI?usp=sharing

# Dependencies

Mainly based on Pytorch 2.0.1

# Model training and sampling

We provide examples of EDM training, consistency distillation, consistency training, single-step generation, and model evaluation in [launch.sh](launch.sh).

# Zero-shot editing
We provide examples for multistep generation and zero-shot image editing in [editing_multistep_sampling.ipynb](editing_multistep_sampling.ipynb).

# Citation

If you find this method and/or code useful, please consider citing

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023},
}
```

This repo is built upon previous work [score_sde](https://github.com/yang-song/score_sde). Please consider citing

```bibtex
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
