# tensorboard

```shell
(tf_12) xxi@xxi-B85M-DS3H ~/Desktop/tf_practice/mnist (master)$ ls
1.mnist_softmax.py       4.mine_mnist_conv.py    pics
2.mine_mnist_softmax.py  5.mnist_tensorboard.py  README.md
3.mine_mnist_mlp.py      logs # <= log folder
```
```shell
(tf_12) xxi@xxi-B85M-DS3H ~/Desktop/tf_practice/mnist (master)$ tensorboard --logdir='logs/' # run
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Starting TensorBoard 39 on port 6006
(You can navigate to http://127.0.1.1:6006) # copy link paste to chrome

```
