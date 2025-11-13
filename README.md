# Open Perceiver

Perceiver is a modifcation to the Transformer Decoder to flexibly handle inputs with large size. In this repository we replicate end-to-end trianing on ImageNet.

## Setup

```
pip install -r requirements.txt
```

## Training on ImageNet

You can use this [data download script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) to download ImageNet.

**Training scaled-down version**

```
python train.py fit -c config_imagenet_small.yaml
```

After 55h on a single 8xA100 (40GB) node I am able to match the paper result for the scaled-down perciever of 70% accuracy (see figure 5).

**Training full version**

```
python train.py fit -c config_imagenet.yaml
```

After about 12 days on a single 8xA100 (40GB) node I am getting 74.6% accuracy, which is a bit over 3% short than the result from the paper.

## References

* The Perceiver [paper](https://arxiv.org/abs/2103.03206) by Andrew Jaegle and others. Very easy to follow, highly recommend reading through. Also recommend [this talk](https://www.youtube.com/watch?v=wTZ3o36lXoQ) given at Stanford by Andrew Jaegle.
* The [Model definition](https://github.com/lucidrains/perceiver-pytorch) from Phil Wang is really useful as a starting point, but his implementation does deviate from the paper in a couple of ways, so my implementation is slightly different.
* Yannic Kilcher has a nice [explanation](https://www.youtube.com/watch?v=P_xeshTnPZg).