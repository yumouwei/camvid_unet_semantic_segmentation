# Multiclass semantic segmentation of CamVid dataset using U-Net

![U-Net 0001TP_2](https://github.com/yumouwei/camvid_unet_semantic_segmentation/raw/main/images/u-net_0001TP_2.gif)
<br>_Test sequence 0001TP_2. Left to right: input image sequence, true masks, and predicted masks overlaid onto the images_

This repo consists of my implementation of the U-Net model and two additional variants using a pre-trained ResNet50V2 or MobileNetV2 for performing semantic segmentation for the _Cambridge-driving Labeled Video Database (CamVid)_. I implemented the model in `tensorflow==2.11`.  Please use `./train_model.ipynb` to train new models and `./evaluate_model.ipynb` to evaluate the performances.

I treated this project as an opportunity to learn the image segmentation task, including how to prepare the data, what are some of the popular models, and how to evaluate their performances. For this reason I chose a simple algorithm (i.e. U-Net) and a small enough dataset so that I can run these on my own hardware. My final models are still far from state-of-the-art performance but I'm still impressed by how well they work given their simple implementations. I also chose not to include the final model in this repo (even though I kept the `./models` folder) because of their sizes (300~600 MB each) and how easy they can be trained on basic hardware or on Colab.

Some of the useful papers and links I referred to include:
- Overview of image segmentation: https://arxiv.org/abs/1704.06857, https://arxiv.org/abs/2001.05566, https://www.jeremyjordan.me/semantic-segmentation/, https://medium.com/swlh/image-segmentation-using-deep-learning-a-survey-e37e0f0a1489
- The original U-Net paper: https://arxiv.org/abs/1505.04597
- The SegNet paper, which includes a benchmark on the CamVid dataset: https://arxiv.org/abs/1505.07293
- Papers describing the CamVid dataset: https://doi.org/10.1016/j.patrec.2008.04.005, https://www.robots.ox.ac.uk/~lubor/bmvc09.pdf
- Benchmarks: https://paperswithcode.com/sota/semantic-segmentation-on-camvid

## 1. Semantic segmentation using U-Net
- Semantic segmentation: assign label to every pixel; single or multiclass
- Differs from instance segmentation (also distinguishes each individual object/instance) and object detection (find bounding box of each object)
- U-Net, one of the easiest model, originally used for medical physics (or biology?) but has been applied to many other applications
- Architecture: fully convolutional autoencoder with residual/skip connections
- Training: image to masks (indexed, different value for each class)
- Metrics (commonly quoted): 1. global pixel accuracy, 2. class pixel accuracy, 3. mIOU/Jaccard score

## 2. Dataset

- Describe CamVid dataset: sequences, data split, 32 classes
- 11 categories (often quoted in studies such as the SegNet paper). Available at here (https://github.com/lih627/CamVid)

## 3. Implementations in tf.keras
- Used which version of tensorflow 
- Network: convolutional block (Conv2D-BN-GeLU-Conv2D-BN-GeLU), pooling and TransposeConv2D for downsampling/upsampling at each stage. 
- (In constrast, SegNet uses pooling indices for upsampling)
- Variants using pre-trained encoders

## 4. Results

- Show images (image, true_mask, pred_mask)
- Compare 3 U-Net models -- hypothesis on why the pre-trained encoder models don't work better (dataset too small)
- Compare to SegNet results - "result leaves a lot to be desired" -- reported SegNet result also used additional training data

## 5. Discussion and future works
- Data augmentation -- surprisingly difficult with TF2
- Alternative dataset - e.g. CityScapes & the other one mentioned in SegNet paper
- Alternative/additional algorithms - e.g. CRF-RNN (can't find functional keras implementation), DeepLab type architecture
