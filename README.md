# Multiclass semantic segmentation of CamVid dataset using U-Net
<div>
<img src="https://github.com/yumouwei/camvid_unet_semantic_segmentation/raw/main/animations/u-net_0001TP_2.gif" width="800" >
</div>

**_Test sequence 0001TP_2. Left to right: input image sequence, true masks, and predicted masks overlaid onto the images_**


This repo consists of my implementation of the U-Net model and two additional variants using a pre-trained ResNet50V2 or MobileNetV2 for performing semantic segmentation for the _Cambridge-driving Labeled Video Database (CamVid)_. I implemented the model in `tensorflow==2.11`.  Please use `./train_model.ipynb` to train new models and `./evaluate_model.ipynb` to evaluate the performances.

I treated this project as an opportunity to learn the image segmentation task, including how to prepare the data, what are some of the popular models, and how to evaluate their performances. For this reason I chose a simple algorithm (i.e. U-Net) and a small enough dataset so that I can run these on my own hardware. My final models are still far from state-of-the-art performance but I'm still impressed by how well they work given their simple implementations. I also chose not to include the trained models in this repo (even though I kept the `./models` folder) because of their sizes (between 300~600 MB) and how easy they can be trained on basic hardwares or on Colab.

Some of the useful papers and links I referred to include:
- Overview of image segmentation: https://arxiv.org/abs/1704.06857, https://arxiv.org/abs/2001.05566, https://www.jeremyjordan.me/semantic-segmentation/, https://medium.com/swlh/image-segmentation-using-deep-learning-a-survey-e37e0f0a1489
- The original U-Net paper: https://arxiv.org/abs/1505.04597
- The SegNet paper, which includes a benchmark on the CamVid dataset: https://arxiv.org/abs/1505.07293
- Papers describing the CamVid dataset: https://doi.org/10.1016/j.patrec.2008.04.005, https://www.robots.ox.ac.uk/~lubor/bmvc09.pdf
- Benchmarks: https://paperswithcode.com/sota/semantic-segmentation-on-camvid

## 1. Semantic segmentation using U-Net

The goal of semantic segmentation is to assign a label to every pixel in a image. It differs from object detection, which tries to find a bounding box for each of the detected object, and instance segmentation, which tries to assign both the semantic class and the specific object instance to each pixel. The class labels can either be binary ("doesn this pixel belong to the class or not") or multiclass ("which class does this label belongs to"). For this project I focused on multiclass semantic segmentation.

One of the simpliest models for semantic segmentation is the U-Net. This network basically consists of a symmetric fully convolutional encoder-decoder network with skip connections between each encoder-decoder stage. It was originally created for segmenting biomedical images but has been applied to many other areas (probably because how easy it can be implemented). The model is trained to reproduce the given segmented masks using the input images.

Some common metrics for evaluating a model's performance includes:

1. Pixel-wise prediction accuracy per class ("how many pixels belonging to a specific class are predicted correctly"), as well as the average across all classes (excluding "Void").
2. Global pixel-wise prediction accuracy ("how many pixels, regardless of class distinctions ("Void" excluded), are predicted correctly")
3. Intersection-over-union (IOU), also known as the Jaccard score, per class ("how does the prediction overlap with the ground truth"), and the average across all classes or mIOU.

Please refer to the review papers for the definition of each metric. The functions for evaluating these metrics are stored in `./utils.py`.

## 2. Dataset
<div>
<img src="https://user-images.githubusercontent.com/46117079/235361740-4f6a6607-d2a7-48ff-893d-ed5c5226bdb9.png" width="600" >
</div>

**_11 semantic categories and data splits_**

CamVid is a database for understanding road or driving scenes. It consists of 701 images and hand-annotated masks captured from 5 driving video sequences. The original dataset contains 32 semantic classes, although a 11-category classification (which combines several similar classes) is more often used in the literatures. The 701 image-mask pairs are split into 367 for training, 101 for validation, and 233 for testing.

The data I used (which are included in the `./data` folder) come from [this repo](https://github.com/lih627/CamVid) credited to [lih627](https://github.com/lih627).

## 3. Implementations in tensorflow

My implementation of the U-Net neural network is available in `./build_model.py`. The model consists of 4 encoder and decoder stages with a latent stage in between. Each stage consists of a convolution block plus a MaxPooling2D or Conv2DTranspose layer for down/upsampling. For the convolution block I used 2x(Conv2D-BatchNorm-GeLU). From my tests this performed the best compared to other architectures by either swapping the GeLU with ReLU activation, by dropping the BatchNorm layers, or by replacing them with Dropouts.

I resized the images & masks to 224x224. The images have the shape (224, 224, 3), 3 for each of the RGB channels. The masks have the shape (224, 224, 1), the last dimension being the integer category label ranging from 0~10 and 255 (the 'Void' class, which is ignored in loss calculation). I used the SparseCategoricalCrossentropy loss and Adam optimizer. One could also use CategoricalCrossentropy loss but that would require one-hot encode the masks and be very memory-intensive. 

Besides the vanilla U-Net I also implemented 2 modified models using the feature extractors from pre-trained ResNet50V2 and MobileNetV2. The decoder networks used in these 2 models are identical to the vanilla U-Net.

## 4. Results
<div>
<img src="https://user-images.githubusercontent.com/46117079/235374354-f75ce977-163b-4cd6-929e-6f0700f44a55.png" width="1000" >
</div>

**_Image, true mask, predicted mask using the vanilla U-Net_**


| **Model** | U-Net | w.ResNet50V2 | w.MobileNetV2 | _*SegNet (3.5k training set)_ |
|---|---|---|---|---|
| _Sky_ | **96.1** | 95.7 | 94.6 | _96.1_ |
| _Building_ | 85.3 | 84.0 | **90.0** | _89.6_ |
| _Pole_ | 22.0 | **28.4** | 17.4 | _32.1_ |
| _Road_ | **96.7** | 96.5 | 94.3 | _96.4_ |
| _Pavement_ | 80.5 | 85.9 | **89.2** | _62.2_ |
| _Tree_ | 76.7 | 75.1 | **79.0** | _83.4_ |
| _SignSymbol_ | 47.3 | **50.3** | 29.9 | _52.7_ |
| _Fence_ | 23.3 | **31.5** | 27.5 | _53.45_ |
| _Car_ | **81.6** | 74.1 | 72.5 | _87.7_ |
| _Pedestrian_ | **60.6** | 41.1 | 37.4 | _62.2_ |
| _Bicyclist_ | **32.0** | 29.4 | 11.9 | _36.5_ |
| **Class avg.** | **58.5** | 57.7 | 53.6 | _71.20_ |
| **Global Avg.** | 86.4 | 86.0 | **87.0** | _90.40_ |
| **mIOU** | **0.52** | 0.50 | 0.50 | _0.60_ |

**_Pixel accuracy by class, class average, global average, and mIOU of the 3 trained U-Net models, plus SegNet (reported from the [paper](https://arxiv.org/abs/1505.07293)). The SegNet model was also trained on an extended 3.5k image training set so it isn't exactly a fair comparison._**


<div>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/46117079/235374182-2c0e003e-e0cf-41f4-8c37-8e63bf0524da.png">
</div>

**_Pixel accuracy vs IOU by class_**


<div>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/46117079/235381261-22905417-48a3-4f81-a306-37a6133fccfc.png"></div>

**_Pixel accuracy vs percentage in test set data_**

The results shown above definitely leaves a lot to be desired. While the models do decently well for classes that have more pixels in the training data such as Sky, Road, Building and to some extent Car and Tree, they perform poorly for smaller classes such as Pole, Fence, and Bicyclist. If this is going to be used for a self-driving system I'd be worried if the model cannot identify a road sign not to say hitting a cyclist or a pedestrian. As a cyclist myself I'm well aware how dangerous those cars and trucks can be when their drivers aren't aware of the cyclist near their vehicles.

I included the SegNet benchmarks as I can't find one for U-Net. The SegNet is a very similar model, except it uses pooling indices instead of trainable convolutional layers for upsampling. I'm quite surprised that my models actually do better in 3 of the 11 categories, especially since the SegNet model was trained on a significantly larger dataset.

I'm also surprised that the two other models using pre-trained models didn't perform much better than the vanilla U-Net. My hypothesis is that -- because of how small the training dataset is, the model does not need a more sophisticated encoder network (or perhaps we can call it "transferable knowledge?") to capture the patterns in this dataset. With a bigger dataset maybe these two models would improve more compared to the vanilla model. To be fair it could also be because of the decoder networks which are basically the same (except at the concatenate layers) for each of the models.

Finally, I've read about arguments for or against using different metrics -- whether pixel accuracy or IOU or something else is better for evaluating a  segmentation model. From my tests it seems there is a decent correlation between class pixel accuracy and class IOU, except the anomalies of SignSymbol and Pedestrian. But anyway at least I can tell a good model is a good model using either of these two metrics.

## 5. Future works

Here are some of the ideas for improving the model I've either thought about, read about but haven't tried, or something I've tried but didn't work out:

1. Use data augmentation. This was emphasized in the original U-Net paper as they had a very limited dataset for training their model. I thought this was going to be a easy thing to do but it turned out to be quite difficult using the augmentation functions in `tf.keras`.
2. Add more training data, such as the Cityscapes dataset which is a significantly bigger dataset than CamVid. From my experience add more data almost always help improving the performance of a model.
3. Add a Conditional Random Fields module (such as a CRF-RNN layer) after the neural network model. This was used in the first version of DeepLab. I found [this implementation of CRF-RNN for the original keras](https://github.com/sadeepj/crfasrnn_keras) but I couldn't get it to work with tensorflow 2. That author does have an alternative implementation for pytorch which has a lot better documentation than the keras version.
4. (Obviously) try other algorithms.
