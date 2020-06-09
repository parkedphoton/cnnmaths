# cnnmaths
Here is some Python code to explore the maths behind the convolution layers in a Convolutional Neural Network (CNN)



## Install

```
$ mkvirtualenv cnnmaths -p python3
$ pip install -r requirements.txt
```

pytorch needs to be installed separately, see
[install Pytorch](https://pytorch.org/)



## Run

compare the following:
```
$ python learn_filters.py
$ python learn_filters_pytorch.py
```

The first one is implemented using ad-hoc filter/convolution operations
the second one is implemented using pytorch

Here is the maths behind the first implementation:
[The Maths of Convolution in CNN](https://parkedphoton.com/the-maths-of-convolution-in-cnn/)





