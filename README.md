# Multiclass semantic segmentation of CamVid dataset using U-Net
<div>
<img src="https://github.com/yumouwei/camvid_unet_semantic_segmentation/raw/main/images/u-net_0001TP_2.gif" width="600" >
</div>

**Test sequence 0001TP_2. Left to right: input image sequence, true masks, and predicted masks overlaid onto the images**


This repo consists of my implementation of the U-Net model and two additional variants using a pre-trained ResNet50V2 or MobileNetV2 for performing semantic segmentation for the _Cambridge-driving Labeled Video Database (CamVid)_. I implemented the model in `tensorflow==2.11`.  Please use `./train_model.ipynb` to train new models and `./evaluate_model.ipynb` to evaluate the performances.

I treated this project as an opportunity to learn the image segmentation task, including how to prepare the data, what are some of the popular models, and how to evaluate their performances. For this reason I chose a simple algorithm (i.e. U-Net) and a small enough dataset so that I can run these on my own hardware. My final models are still far from state-of-the-art performance but I'm still impressed by how well they work given their simple implementations. I also chose not to include the trained models in this repo (even though I kept the `./models` folder) because of their sizes (between 300~600 MB) and how easy they can be trained on basic hardwares or on Colab.

Some of the useful papers and links I referred to include:
- Overview of image segmentation: https://arxiv.org/abs/1704.06857, https://arxiv.org/abs/2001.05566, https://www.jeremyjordan.me/semantic-segmentation/, https://medium.com/swlh/image-segmentation-using-deep-learning-a-survey-e37e0f0a1489
- The original U-Net paper: https://arxiv.org/abs/1505.04597
- The SegNet paper, which includes a benchmark on the CamVid dataset: https://arxiv.org/abs/1505.07293
- Papers describing the CamVid dataset: https://doi.org/10.1016/j.patrec.2008.04.005, https://www.robots.ox.ac.uk/~lubor/bmvc09.pdf
- Benchmarks: https://paperswithcode.com/sota/semantic-segmentation-on-camvid

## 1. Semantic segmentation using U-Net

The goal of semantic segmentation is to assign a label to every pixel in a image. It differs from the task of object detection, which tries to find a bounding box for each of the detected object, and instance segmentation, which tries to assign both the semantic class and the specific object instance to each pixel. The class labels can either be binary ("Is this pixel part of this object class or not") or multiclass ("which class does this label belongs to"). For this project I focused on multiclass semantic segmentation.

One of the simpliest models for semantic segmentation is the U-Net, which basically consists of a symmetric fully convolutional encoder-decoder network with skip connections between each encoder-decoder stage. It was originally used for segmenting biomedical images but has been applied to many different areas (probably because how easy it can be implemented). The model is trained to reproduce the given segmented masks using the input images.

Some common metrics for evaluating a model's performance includes:

1. Global pixel-wise accuracy
2. Pixel-wise accuracy per class, as well as the class-average
3. Intersection-over-union (IOU, also known as the Jaccard score) per class, and the class-average (mIOU)

Please refer to the review papers for the definition of each metric. The functions for evaluating these metrics are stored in `./utils.py`.

## 2. Dataset
<div>
<img src="https://user-images.githubusercontent.com/46117079/235361740-4f6a6607-d2a7-48ff-893d-ed5c5226bdb9.png" width="600" >
</div>

**11 semantic categories and data splits**

The CamVid database for road/driving scene understanding consists of 701 images and hand-annotated masks captured from 5 driving video sequences. The original dataset contains 32 semantic classes, although a 11-category classification (which combines several similar classes) is more often used in literatures. The 701 image-mask pairs are split into 367 for training, 101 for validation, and 233 for testing.

The data I used (which are included in the `./data` folder) comes from [this repo](https://github.com/lih627/CamVid) credited to [lih627](https://github.com/lih627).

## 3. Implementations in tensorflow

My implementation of the U-Net neural network is available in `./build_model.py`. The model consists of 4 encoder and decoder stages with a latent stage in between. Each stage consists of a convolution block and a MaxPooling2D or Conv2DTranspose layer for downsampling or upsampling. For the convolution block I used 2x(Conv2D-BatchNorm-GeLU); from my tests this performed the best compared to other architectures by either swapping the GeLU with ReLU activation, by dropping the BatchNorm layers, or by replacing them with Dropouts.

I resized the images & masks to 224x224. The images have the shape (224, 224, 3), 3 for each of the RGB channels. The masks have the shape (224, 224, 1), the last dimension being the integer category label ranging from 0~10 and 255 (the 'Void' class, which is ignored in loss calculation). I used the SparseCategoricalCrossentropy loss and Adam optimizer. One could also use CategoricalCrossentropy loss but that would require one-hot encode the masks and be very memory-intensive. 

Besides the vanilla U-Net I also implemented 2 modified models using the feature extractors from pre-trained ResNet50V2 and MobileNetV2. The decoder networks used in these 2 models are identical to the vanilla U-Net.

## 4. Results

- Show images (image, true_mask, pred_mask)
- Compare 3 U-Net models -- hypothesis on why the pre-trained encoder models don't work better (dataset too small)
- Compare to SegNet results - "result leaves a lot to be desired" -- reported SegNet result also used additional training data

## 5. Discussion and future works
- Data augmentation -- surprisingly difficult with TF2
- Alternative dataset - e.g. CityScapes & the other one mentioned in SegNet paper
- Alternative/additional algorithms - e.g. CRF-RNN (can't find functional keras implementation), DeepLab type architecture
