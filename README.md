# Maestro: Uncovering Low-Rank Structures via Trainable Decomposition

Implementation of Maestro's LOD (Low-rank ordered Decomposition) for different type of NN layers in PyTorch.

The repo contains:

- layers: [Linear](maestro/layers/linear.py), [Convolution](maestro/layers/conv.py), [Transformers](maestro/layers/transformer.py)
- Ordered Dropout (OD) Samplers: [One layer at time](maestro/samplers/single_layer.py) (used in our work), [Per Layer Independent Sampling](maestro/samplers/independent.py), [PufferFish Sampling](maestro/samplers/pufferfish_sampler.py),
- Examples:
  -  Linear models that recover PCA. SVD and correct importance ordering [linear_models.ipynb](linear_models.ipynb),
  -  CIFAR10 + MNIST experiments [cifar_mnist_main.py](cifar_mnist_main.py), bash [run_cifar_mnist.sh](run_cifar_mnist.sh),
  -  ImageNet experiments [imagenet_main.py](imagenet_main.py), bash [run_imagenet.sh](run_imagenet.sh).

### How to run

```
pip install -r requirements
python setup.py develop
```

### Reference

If you find this repo useful, please cite the paper:

```
@inproceedings{
    horv{\'a}th2024maestro,
    title={Maestro: Uncovering Low-Rank Structures via Trainable Decomposition},
    author={Samuel Horv{\'a}th and Stefanos Laskaridis and Shashank Rajput and Hongyi Wang},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=7bjyambg4x}
}
```
