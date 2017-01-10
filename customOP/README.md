``` shell
xxi@xxi-B85M-DS3H ~/Desktop/tf_practice/customOP $ ls    [ruby-2.3.1p112]
make.sh  test_zero.py  WORKSPACE  zero_out.cc  zero_out_op_test.py  zero_out.so
```

After modify `*.cc` or `*.cu.cc` **MUST** run `bash make.sh` to create new `*.o` and `*.so`
`*.o` creat by `nvcc`

Reference of [make file](https://github.com/smallcorgi/Faster-RCNN_TF/blob/master/lib/make.sh#L12-L19)
**NOTE** If compile New OP with `gcc5` need to add `-D_GLIBCXX_USE_CXX11_ABI=0` to `g++ -std ...` (already add)    
If compile with `gcc4` never mind

```shell
(tf_12) xxi@xxi-B85M-DS3H ~/Desktop/tf_practice/customOP $ bash make.sh
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
```
