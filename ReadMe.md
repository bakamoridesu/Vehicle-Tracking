# Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform  feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## 1. Dataset exploration

The dataset contains 8792 vehicle images and 8968 non-vehicle images with the shape of (64, 64, 3) each.

Here are some examples of the images:

![Gray image example](/images/vehicle_images.png)
![Gray image example](/images/non_vehicle_images.png)

## 2. Feature extraction.

### 2.1. Spatial binning.

For the one of the features I performed spatial binning on an image to retain information to help in finding vehicles.
First I had resized an image to smaller size yet the car was still recognizeable. Then I had flattened the array of the raw pixels of a resulted image. 

Here is how do the binning features look like for vehicles and non-vehicles:

![Gray image example](/images/vehicle_spat.png)
![Gray image example](/images/non_vehicle_spat.png)

The car histograms seem to have roughly similar forms, and the histograms of all other road stuff vary a lot. But I want to take a closer look at typical car image and histogram in comparison with typical non-car.

![Gray image example](/images/single_vehicle_spat.png)
![Gray image example](/images/single_vehicle_hist.png)

So the histogram of the typical back of a car looks like exponential decay, and the road image's histograms are different but mostly they look more like normal distribution. This feature seems to be useful for car detection.

### 2.2. Histograms of color.

Now I'll look at histograms of pixel intensity (color histograms) as features.

The idea is to take feature vectors for all 3 color channels. Here is how the single image vectors look like:

![Gray image example](/images/color_hist.png)

### 2.3. HOG features.

Taking HOG a.k.a. Histogram Of Oriented gradients is a great way to extract features from an image. To extract them, I used `skimage`'s `hog` function.

![Gray image example](/images/HOG_features.png)

It seems like HOG feature represents the shape of an object. And gets rid of most of the noise. Very useful.

So I decided to use all of the described features for classification. To do it, I just extract all features vectors and then concatenate them for each image.

## 3. Train classifier.

To actually choose the parameters and the color scheme to feed to classifier, I decided to run the classifier with different parameters directly and look how will they work. The classifier itself can tell what parameters work better. 

First I just ran a default SVM classifier with some default parameters for feature extraction. The images were in 'RGB' color space. And I got an accuracy of 95.5% on a test set
```
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 7872
43.87 Seconds to train SVC...
Test Accuracy of SVC =  0.9552
```

Then I tried to use different color spaces for classification:

```
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 7872

Color space = HSV
Test Accuracy of SVC =  0.9778

Color space = HLS
Test Accuracy of SVC =  0.978

Color space = YUV
Test Accuracy of SVC =  0.9851

Color space = YCrCb
Test Accuracy of SVC =  0.9817
```

In this experiment YUV color scheme have shown the best results. But what if some of color schemes are better for different features? I wanted to make some tests.

#### using only spatial bin features for classification. trying different color schemes:

```
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 768

Color space = RGB
Test Accuracy of SVC =  0.9127

Color space = HSV
Test Accuracy of SVC =  0.9051

Color space = HLS
Test Accuracy of SVC =  0.8902

Color space = YUV
Test Accuracy of SVC =  0.9285

Color space = YCrCb
Test Accuracy of SVC =  0.9305
```

#### using only color features

```
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 48

Color space = RGB
Test Accuracy of SVC =  0.4958

Color space = HSV
Test Accuracy of SVC =  0.8238

Color space = HLS
Test Accuracy of SVC =  0.8269

Color space = YUV
Test Accuracy of SVC =  0.5206

Color space = YCrCb
Test Accuracy of SVC =  0.5051
```

#### using only hog features

```
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 7056

Color space = RGB
Test Accuracy of SVC =  0.9274

Color space = HSV
Test Accuracy of SVC =  0.9479

Color space = HLS
Test Accuracy of SVC =  0.9519

Color space = YUV
Test Accuracy of SVC =  0.9578

Color space = YCrCb
Test Accuracy of SVC =  0.9555
```
So the YUV and YCrCb formats are almost senseless to color features. But the HLS format is good for it. Also the YUV format seems to be good for HOG features extraction. 

Then I found the right color schemes for all types of features, I tried to classify images varying parameters like number of orientations and pixels per cell. And finally I tried different classifiers. The best classifier was an `MLPClassifier`(multi-layer perceptron) from `sklearn.neural_network` package.

```
20.15 Seconds to train MLP...
Test Accuracy of MLP =  0.9969
```
So with an accuracy of 99.69% the confusion matrix looks like this:

![Gray image example](/images/matrix.png)

Seems that without additional tricks there should not be a lot of false-positives.

## 4. Detection pipeline.

For the single image vehicle detection I implemented a sliding-window algorithm. There is a small size window sliding along the X and Y axis of an image. For every window position the algorithm extracts all the features and the classifier predicts if there is a car detected. If the car is detected within a window, the algorithm draws a box around the window. I use two window sizes, passing them as a scale of the default size. So if the default size is 64 and the scale is 1.5, the window size becomes 64x1.5 = 96 pixels for width and height.

![Gray image example](/images/result_2win.png)

Sometimes the false-positives can be found. To remove them I use threshold for the boxes found. So if the threshold equals 1, some random false-positives represented by one box will be removed. To implement thresholding I use heatmaps, adding 1 to all the pixels fenced by a box. So if two boxes are overlapping, the pixels in the region of overlap will have the value of 2, the non-overlapped pixels will have value of 1 and all other pixel values will equal zero.

![Gray image example](/images/heatmap.png)

Then, to draw box around the heatmap, I use `label` function from `scipy.ndimage.measurements`

![Gray image example](/images/boxes.png)

For the video pipeline I don't use a single image threshold but instead I use a threshold for previous states. If the car was not presented on previous frames and was found on the current frame, the algorithm considers it as an accidentally found false-positive.
The final video can be found here: https://github.com/bakamoridesu/Vehicle-Tracking/blob/master/project_video.mp4

---

### Discussion

At this point the algorithm is doing OK. But it needs some improvements in execution speed and detection accuracy. For the former I want to try taking HOG features from the whole image and then slide along the HOG cells. I've already added some TODOs in the code to start working on it. Will do later. For the latter I want to try using more window sizes and applying threshold for every size. But it's possible only improving the execution speed.

