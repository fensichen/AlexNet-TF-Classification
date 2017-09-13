# AlexNet-TF

This is a python implementation of AlexNet in TensorFlow. The work has described at [ImageNet Classification with Deep Convolutional Neural Networks
Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)


# Usage
python main.py

Please provide the path to train.txt and test.txt to main.py. Each of them list the complite path to your train/val images together with the class number in the following structure.

Example train.txt:
/path/to/train/image1.png 0

/path/to/train/image2.png 1

/path/to/train/image3.png 2

/path/to/train/image4.png 0


were the first column is the path and the second the class label.

### Requirements
	* python 2.7
	* tensorflow 1.3
	* numpy
	* scipy


# Content
	* data.py: Class with data loading and augmentation
	* main.py: Class with graph definition of AlexNet 
	* (finetune): to be added

## Note 
	The LRN layers have been removed.

