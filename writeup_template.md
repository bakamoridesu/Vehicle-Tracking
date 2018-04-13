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


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

