# Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform  feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


### 1. Dataset exploration

The dataset contains 8792 vehicle images and 8968 non-vehicle images with the shape of (64, 64, 3).

Here are some examples of the images:

![Gray image example](/images/vehicle_images.png)
![Gray image example](/images/non_vehicle_images.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

