# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data_dist.png
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./balanced_training_set.png
[image4]: ./new-sign2.jpg
[image5]: ./new-sign3.jpg
[image6]: ./new-sign4.jpg
[image7]: ./new-sign5.jpg
[image8]: ./new-sign6.jpg
[image9]: ./new-sign7.jpg
[image10]: ./new-sign8.jpg
[image11]: ./new-sign9.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data samples are distributed in the training set. I also plotted it for validation and test set, you can check up my .ipynb file.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it proves to have a positive impact on increasing model accuracy. Also it reduced the channels from 3 to 1, meaning fewer parameters and operations for training model.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As the second step, I normalized the image data to zero mean and unit standard deviation because it make all the input data features to fall in to a small range. This makes all the weights update in a similar step size. In theory this can help the training converge sooner.

As a last step, I randomly shuffled data because introducing randomness can better generalize the model.

### Notes on Data Augmentation

I decided to generate additional data because with the model trained on the original gray scaled data  I noticed that classes with more samples in the training set all get good loss and prediction accuracy. Some but not all classes with fewer samples perform significantly worse.

#### random rotation and shifting plus data balancing

To add more data to the the data set, I used the `rotate()` and `shift()` from `scipy` library. A sign tilted a bit or on the edge of image is still the same sign.

I also want to balance the data set so I generated more fake data for classes with fewer sample.

The classes with most samples have arond 2000 samples. But I chose to make all classes have around 6000 samples. Because if I chose 2000, there would be no augmented data for those classes with more samples. That is another kind of unbalance. We can generate more, then reduce the samples uniformly for all classes later if need be.

After you add newly generated data to the original data, you would see the following histogram charts.

![alt_text][image3]

Not perfect, but we can extract the first say 3000 samples for each class and form a new training set based on that. Then it'll be perfectly balanced.

**Question Alert**

By shifting the data I moved some signs to the edges of the image. Rotation and shifting create black area on the edges. So after I did normalization some signs might be damaged. Is this bad? If so, how to address this?

Unfortunately, augmented data doesn't work very well in my case. I'll talk about it next.

#### Underfitting and Overfitting. Most important problem for me in this project, please read!
Instead of underfitting and overfitting, I measure my model by the size of **the gap** between training and validation accuracy and the **numbers they plateau** at. We want the gap to be as small as possible and the numbers they plateau at to be as big as possible.

The numbers they plateau at can be controlled by regularization. In my case I used **Dropout**.

**The real problem is the gap between training and validation accuracy. If I use deeper model or apply augmented data the gap size increases. Can't find a way to address this so I have to use vanilla LeNet-5 model. The optimal gap size I can get is 4%.**

Other students seemed to have done better with augmented data. I wonder what I missed.

But one thing I observed in training on augmented data is with the same dropout probability it was underfitting. That's expected because data augmentation is one kind of regularization.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, no padding, outputs 28x28x6 	    |
| Dropout               | randomly set some activation maps from        |
|                       | previous layer to zero                        |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, no padding, outputs 10x10x16      |							
| Dropout               | same as before                                |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Flatten               | outputs 400                                   |
| Fully connected		| output 120        							|
| Dropout               | randomly drop some nuerons of 120 inputs      |
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| Dropout               | same as before                                |
| RELU                  |                                               |
| Fully connected       | output 43                                     |        			
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The optimizer is called ADAM. It's a combination of Momentum and AdaGrad. Momentum can metigate the problem of being stuck at local minima or saddle point. AdaGrad can metigate the problem of going zig-zag towards the minima. Both are techniques used to converge sooner.

batch size, epoches and learning rate are 128, 50 and 0.001. I tried exponential decay for learning rate, not very useful in this case.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.952
* test set accuracy of 0.934

* Architecture

I tried deeper model with good characteristics in theory. Like two 3x3 layers instead of a 5x5, so fewer weights in total while they are basically equivalent in fact if you think about the effective receptive field size the 2nd 3x3 has on the inputs to the 1st 3x3. Padding the activations to keep the pixels on the edges around in the network depth-wise.

But it has the increasing gap probelm. So I didn't end up with using it.

* Data Augmentation
The only useful augmentation is turning RGB images to gray scale. It bumps the result by 2% on the vanilla LeNet-5 without regularization.

* Hyper Parameters
The only useful change is increasing EPOCHs

* Regularization
I used Dropout. Tuning it does have an positive impact on the result. But it didn't seem to have an impact on gap size in my case.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

The fifth image might be difficult to classify because it is tilted and not in the center of the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			               |     Prediction	        					   | 
|:----------------------------:|:---------------------------------------------:| 
| Yield      		           | Road Work  								   | 
| Speed Limit 80km/h           | Speed Limit 70km/h 						   |
| No Passing			       | No Passing									   |	
| Stop	      		           | Turn Left Ahead					 		   |
| Children Crossing		       | Roundabout mandatory      					   |
| Dangerous curve to the right | Dangerous curve to the right                  |
| No entry                     | No entry                                      |
| Keep right                   | Keep right                                    | 


The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

the model is not certain about the 2nd and the 5th image. The correct class for the 5th image is not on the top 5 softmax list. I mentiond the possible cause a few sections before. As for other wrongly classified images, I don't have good guess. But in previously generated models they did get classified correctly. Maybe something can be done when downsampling the images so they are more recognizable.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.         			| Road Work   									| 
| .38     				| Speed limit 70km/h 							|			
| .72					| No Passing									|
| .89	      			| Turn Left Ahead					 			|
| .58				    | Roundabout mandatory      					|
| .79                   | Dangerous curve to the right                  |
| 1.                    | No entry                                      |
| 1.                    | Keep right                                    |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In the 1st conv layer it's edges shown in the image. In the 2nd conv layer the network starts to focus on different parts of the image in different activation maps.
