# MNIST camera


## Versions

```
Python 3.8.10
OpenCV 4.0.1
PyTorch 1.9.0
torchvision 0.10.0
```

## 学習 - Trainer
PyTorchのExampleを流用。  
https://github.com/pytorch/examples/tree/master/mnist


## 推論 - Inference

推論結果は、対数の定義に従って確率に変換する。
底はネイピア数2.71828...

https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html



## References

https://github.com/pytorch/examples/tree/master/mnist
https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html