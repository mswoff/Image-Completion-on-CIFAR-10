1. Download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
2. Put the dataset in data/CIFAR-10
3. Create folders data/32x32_CIFAR, data/32x32_CIFAR/test_imgs, data/32x32_CIFAR/train_imgs, data/32x32_CIFAR/valid_imgs
4. Run build_dataset.py
5. Run run.py to train model
  specify model with -m
  specify learning rate with -l
  specify number of epochs to train for with -n
  can restore model from a previous training instance with -r "path to .ckpt"

6. Run predict_images.py to output random images and the model's predictions
  specify model with -m
  specify model saved path with -r "path to .ckpt"
  specify number of image with -n
  specify seed to use for random image generation -s
  specify train or development images with -v
  
7. Run evaluate_model.py to test the models on the test set and output sample predictions
  specify model with -m
  specify model saved path with -r "path to .ckpt"
  specify number of image with -n
  specify seed to use for random image generation -s
