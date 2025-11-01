# Open Perceiver

Train Perceiver on ImageNet. 

Perceiver is a modifcation to the Transformer Decoder to flexibly handle inputs with large size.

Installation:

```
pip install -r requirements.txt
```

Kick off training:

```
python train.py fit -c config_imagenet_small.yaml
```

After 55h on a single 8xA100 (40GB) node I am able to match the paper result for the scaled-down perciever of 70% accuracy.

## References

* [Paper](https://arxiv.org/abs/2103.03206) 
* [Model definition](https://github.com/lucidrains/perceiver-pytorch) from Phil Wang
* ImageNet [data download script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)
* [Explanation](https://www.youtube.com/watch?v=P_xeshTnPZg) by Yannic Kilcher