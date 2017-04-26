#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/pie_chart.jpg "Pie Chart"
[image2]: ./examples/original.jpg "Original Image"
[image3]: ./examples/gray.jpg "Grayscale Image"
[image4]: ./new_images/test13.jpg "Traffic Sign 1"
[image5]: ./new_images/test14.jpg "Traffic Sign 2"
[image6]: ./new_images/test21.jpg "Traffic Sign 3"
[image7]: ./new_images/test25.jpg "Traffic Sign 4"
[image8]: ./new_images/test38.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a pie chart showing the distribution of the classes between training set, validation set, and test set. You can clearly see that the distribution of the classes are similar between the sets, but vary dramatically between classes. Therefore I can speculate that a small class like 37 will not be trained as well as a big class like 38. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in the research paper by Pierre Sermanet and Yann LeCun, they have concluded that grayscale work better than color typically. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

Then I did a expand dimension using numpy because the rgb2gray operation has taken out a dimension. So the output gray training set is shaped: (nx32x32x1) where n is the number of training sets. 

Next I shuffled the data to ensure everytime I run, the network is not learning based on the order of the images.

As a last step, I used sklearn.preprocessing.scale to normalize the data so it has a mean of zero and standard deviation of 1. This is better than the method in the (pixel - 128)/ 128 method because it is less affected by outliers, and does not assume 128 as the average. It also is more robust if the image is not 0-255.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x10 				|
| Convolution 6x6	    | 1x1 stride, valide padding, output 8x8x22		|
| RELU          		| 		       									|
| Max pooling	      	| 2x2 stride,  outputs 4x4x22	 				|
| Fully Connected		| input 352, output 170							|
| ReLU					|												|
| Dropout 				| Training keep prob = 0.75						|
| Fully Connected		| input 170, output 100							|
| Dropout 				| Training keep prob = 0.75						|
| Fully Connected		| input 100, output 43							|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started with the learning rate of 0.001, like in LeNet. It returned good results. I tried a lower value of 0.0005, and the accuracy increased slower and did not reach the same level was reduced after a few epochs. Then I tried a higher learning rate of 0.003. The validation accuracy was high in the first epoch, but had similar or slightly higher accuracy after 10 epochs. It seems that a even higher learning rate wouldn't create much benefit, so I settled on 0.001. I tried increasing the batch size to 256, but that reduced the validation accuracy. I reduced the batch size to 64 and the validation accuracy improved significantly. I did not want to slow down the calculation too much, so I used a batch size of 64. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.958
* test set accuracy of 0.935

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

To train the model, I started out with the basic LeNet lab setup. I thought that the architecture was a good start because traffic signs and hand writing has a lot of similarities, since they are both symbols. I started out normalizing the input using (pixel - 128)/128. I was able to get validation accuracy in the mid to high 80s. After I processed the data using grayscale and sklearn preprocesssing, and added size and depth to my convolution filters, I was able to get validation accuracy up to 90-91%. I saw that the validation accuracy peaked in the first 3-4 epochs, so I thought that the model is probably overfitting, and I need more regulariztion. What made a huge difference was adding a dropout layer with training keep_prob of 0.75. That improved the validation accuracy to 95%. I added a second dropout layer between layer 3-4 to try to reduce overfitting even more. It increased the training speed, and the ratio between validation accuracy and training accuracy improves. I tried adding even more depth to the convolutional layers, but that did not improve the results much, but slowed down the calculation more. So I stuck with 10 and 22 for the depth. 


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]  

The first image might be difficult to classify because the background has a pattern
The second image might be difficult to classify because it is at an angle
The third image might be difficult to classify because it has a complex background
The fourth and fifth image should be pretty easy to classify

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield		      		| Yield   										| 
| Stop Sign    			| Stop Sign										|
| Double Curve			| Children Crossing								|
| Road Work	      		| Road Work						 				|
| Keep Right			| Keep Right		   							|


The model was able to correctly identify 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares similarly to the accuracy on the test set of 0.935, since the sample size is so small. Interestingly, the double curve result had a high softmax confidence that it is a Children crossing sign, and double curve is not in any of the top 5 softmax probabilities.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. This could be because yield is a triangle, and blank in the middle. Those two features are drastically different from the other images. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| yield sign   									| 
| 0.0     				| ahead only 									|
| 0.0					| no vehicles									|
| 0.0         			| priority road					 				|
| 0.0	    		    | turn right ahead     							|


For the second image, the model is relatively sure that this is a stop sign (probability of 0.79), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .79         			| stop sign   									| 
| .13     				| speed limit 80								|
| .06					| speed limit 60								|
| .0079      			| speed limit 50				 				|
| .0036				    | road work         							|

For the third image, the model is relatively sure that this is a children sign (probability of 0.91), and the image does NOT contain a children crossing sign. The correct label of double curve is not even in the top five. It is unclear to my why that is. My hypothesis goes back to the data visualization, where I can see that the portion of the training data set with double curve is quite small. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| children crossing sign   						| 
| .08     				| beware of ice/snow 							|
| .0048					| right of way at next intersection				|
| .00008      			| bike crossing					 				|
| .000007			    | dangerous curve to the right					|

For the fourth image, the model is very sure that this is a road work sign (probability of 0.999), and the image doesThe top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| road work   									| 
| .0     				| double curve 									|
| .0					| speed limit 30								|
| .0         			| speed limit 80				 				|
| .0    			    | right turn ahead    							|

For the fifth image, the model is very sure that this is a keep right sign (probability of 1.0), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| keep right   									| 
| .0     				| slippery road									|
| .0					| speed limit 60								|
| .0        			| wild animals crossing			 				|
| .0				    | speed limit 30      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


