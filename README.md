# Multiclass semantic segmentation of CamVid dataset using U-Net

- Summary of the project

- How to use codes
- Trained model - only provide custom U-Net (other 2 using ResNet50V2 and MobileNetV2 as the encoder don't work better)

- Useful links:


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
- Alternative dataset - e.g. CityScapes & the other one mentioned in SegNet paper
- Alternative/additional algorithms - e.g. CRF-RNN (can't find functional keras implementation), DeepLab type architecture
