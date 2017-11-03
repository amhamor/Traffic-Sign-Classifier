**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class_id_histogram.jpg "Class ID Data Visualization"
[image2]: ./preprocessed_images.png "Preprocessed Images."
[image3]: ./lenet.png "LeNet Architecture."
[image4]: ./german-traffic-sign-test-images/road-narrows-on-the-right-german-traffic-sign.jpg "Road Narrows On the Right German Traffic Sign"
[image5]: ./german-traffic-sign-test-images/end-of-all-speed-and-passing-limits-german-traffic-sign.jpg "End of All Speed and Passing Limits German Traffic Sign"
[image6]: ./german-traffic-sign-test-images/slippery-road-german-traffic-sign.jpg "Snow-Covered Slippery Road German Traffic Sign"
[image7]: ./german-traffic-sign-test-images/priority-road-german-traffic-sign.jpg "Priority Road German Traffic Sign"
[image8]: ./german-traffic-sign-test-images/keep-left-german-traffic-sign.jpg "Keep Left German Traffic Sign"
[image9]: ./blurry-preprocessed-image.png "Blurry Preprocessed Image"
[image10]: ./softmax-probabilities/road-narrows-on-the-right-german-traffic-sign.jpg "Road Narrows On the Right German Traffic Sign Softmax Probabilities"
[image11]: ./softmax-probabilities/end-of-all-speed-and-passing-limits-german-traffic-sign.jpg "End of All Speed and Passing Limits German Traffic Sign Softmax Probabilities"
[image12]: ./softmax-probabilities/slippery-road-german-traffic-sign.jpg "Snow-Covered Slippery Road German Traffic Sign Softmax Probabilities"
[image13]: ./softmax-probabilities/priority-road-german-traffic-sign.jpg "Priority Road German Traffic Sign Softmax Probabilities"
[image14]: ./softmax-probabilities/keep-left-german-traffic-sign.jpg "Keep Left German Traffic Sign Softmax Probabilities"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how my implementation addresses each point.  

---
###Writeup / README
I have created a German traffic sign classifier by training German traffic sign images on a convolution neural network. Preprocessing and structuring of this German traffic sign data into a convolution neural network can be viewed in the following Jupyter Notebook: [German Traffic Sign Classifier](https://github.com/amhamor/Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Dataframe.shape method from the pandas library and built-in Python functions to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 pixels wide by 32 pixels high by 3 pixels deep.
* The number of unique classes/labels in the data set is 43 classes.

####2. Include an exploratory visualization of the dataset.

With matplotlib.pyplot.hist, I can compare the proportion of each class ID example used against each other of the training, validation, and test data sets by plotting a histogram:

![Class ID Histogram][image1]

We can see that the proportion of each class ID example within each of the training, validation, and test is generally even from data set to data set given the scale of the y-axis. Having this generally even proportion of each class ID example helps mitigate the tendency for biased accuracy results between each of the training, validation, and test sets. For example, we might be able to feel confident that a difference in accuracy between the training set and the validation set is not significantly due to proporionately having more of one class ID in the training set compared to the validation set.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried five different methods in preprocessing the data:

	1) Normalize the image pixel values.
	2) Standardize the image pixel values.
	3) Convert the images to grayscale.
	4) Apply gamma correction to the images.
	5) Flip the images horizontally or vertically.

We can see the effects of each method in the following image:

![Preprocessed images.][image2]

While comparing each image to the original image, we can observe at least one comparison for each preprocessed image: 

	1) There appears to be little difference between the original image and normalized image, 
	2) but the standardized image seems to transform low pixel values to pixel values seen as bright colors. 
	3) The grayscale image simply transforms to a black and white image 
	4) while the gamma corrected image has a level of brightness applied to it. 
	5) For the vertically flipped image, we see the image vertically flipped.

When applying each method to training the model, I came to three general conclusions:
	1) Because normalized image pixel values are all positive while standardized image pixel values can be negative, the ReLU activation function can cause data to be lost for any negative pixel values within any standardized image. Thus, I use normalized images over standardized images.
	2) Inputting gamma corrected and horizontally or vertically flipped images cause the model to predict class ID's incorrectly and does not learn. Thus, I avoid using these images.
	3) Using grayscale images prevents the model from learning the colors associated with each German traffic sign, which is a primary characteristic in categorizing traffic signs. Thus, even if grayscale images can help distinguish edges within an image better, I feel that the model loses information that is significantly relevant for traffic sign classification and I avoid using grayscale images.

To reduce overfitting, I attempted to augment the original data set by adding standardized gamma corrected images and flipped images. However, I was unable to start training the model due to memory constraints. I continued utilizing normalized images to reduce the amount of time required for the model to start learning and look for other methods to reduce overfitting.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Normalized RGB image   							| 
| Convolution 5x5     	| 1x1 stride and VALID padding. Output shape = 28x28x12.|
| Max Pool 2x2		| 2x2 stride and SAME padding. Output shape = 14x14x12.	|
| ReLU			|							|
| Local Response Normalization	|						|
| Convolution 5x5	| 1x1 stride and VALID padding. Output shape = 10x10x32.|
| Max Pool 2x2	      	| 2x2 stride and SAME padding. Output shape = 5x5x32.	|
| ReLU			|							|
| Local Response Normalization	|						|
| Convolution 2x2	| 1x1 stride and VALID padding. Output shape = 4x4x64.	|
| Max Pool 2x2		| 2x2 stride and SAME padding. Output shape = 2x2x64.	|
| ReLU			|							|
| Local Response Normalization	|						|
| Fully Connected	| Number of input neurons = 256. Number of output neurons = 120.|
| Fully Connected	| Number of input neurons = 120. Number of output neurons = 84.	|
| Fully Connected	| Number of input neurons = 84. Number of output neurons = 43.	|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

This German traffic sign classifier model calculates the cost using tf.nn.softmax_cross_entropy_with_logits with one-hot encoded labels. This model updates/optimizes weights and biases with tf.train.AdamOptimizer with a learning rate of .002 and minimizing cost. I decided to set the epochs to 10000 to allow the training accuracy to go to 1.00 before I interrupt the training and utilize a batch size of 32. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0000.
* validation set accuracy of 0.9737.
* test set accuracy of 0.9585.

The training set accuracy was calculated immediately after running the optimizer. The validation accurary set accuracy was calculated using the evaluate_accuracy(X_data, y_data, sess) function immediately after calculating the training set accuracy. The test set accuracy was calculated also using the evaluate_accuracy(X_data, y_data, sess) function but in the cell below the cell that trains the model.

I first tried the LeNet architecture:

![LeNet architecture by Yann LeCun.][image3]

This architecture seemed to be effective in predicting handwritten digits from the MNIST dataset so I decided to see how well it worked on the German traffic sign dataset. I observed immediately that the model's accuracy would stay the same for every iteration. Also, I was unsure if two convolutional layers was enough for the model to maximize its detection of the various features within each image. 

After looking at the logits for each iteration, I noticed that the logits would stay the same for every iteration by keeping the number at the same index through each iteration labelled as one and all other numbers as zero after applying a softmax activation function. This is possibly because the weights explode to very large numbers at one indexed location while all others are very small and/or negative. When the model updates its weights, it updates based on how far from one or zero the logits are which is significantly less than the numbers found in the raw logits. This seems to prevent the model from correcting the weights where they are very large or small. Thus, I adjusted the model by returning the logits without applying a softmax activation function.

After removing the softmax activation function, I then noticed that the model would learn at a seemingly decent rate then collapse the accuracy to almost zero. I then observed how the weights changed after each iteration and noticed that they would change to various different numbers at first then decline to almost zero and stay there. I decided that this was due to too high of a learning rate where the model was overstepping the minimum cost value into a local minimum that caused the model's accuracy to remain the same. Thus, I reduced the learning rate from 0.001 to 0.0001. This allowed the model to get up to a validation accuracy of around 0.65.

To increase the validation accuracy, I needed to reduce the overfitting of the model. In attempts to reduce overfitting, I applied dropout to each convolutional layer and added augmented images into the dataset. Both of these resulted in the model failing to learn. After stumbling across the Alexnet architecture, I noticed how a local response normalization function was applied to the model. This provided me the insight needed to realize that because the weights were being allowed to reach large numbers, there may have been less precision in updating the weights. Normalizing the weights after each convolutional layer would ensure that each layer started with numbers on the same scale as all other layers. After applying local response normalization to each convolutional layer, the model learns exponentially faster and reduces overfitting significantly. Along the way, I also realized that using a lower batch size allows the model to update the weights more often and therefore more precisely. Thus, I reduced the batch size from 8192 to 32. The validation accuracy is now able to reach 0.9737 as a result of these adjustments.

To see if the model would detect a higher level of features than the two convolutional layers within the LeNet architecture, I also adjusted the model by adding another convolutional layer. At this point, I am unsure if this added layer has improved the model as this third layer seems to essentially show checkered patterns for each feature within this layer (as I will show at the end of this write up).

Having a validation accuracy of 0.9737 means that the model predicts 4294 out of 4410 images correctly without training on these images. This seems like a reasonable level of prediction for this project and an opportunity to learn what feature can prevent a machine from correctly classifying an image. This brings us to testing the model on various German traffic sign images taken from the internet.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Road Narrows On the Right German Traffic Sign][image4] ![End of All Speed and Passing Limits German Traffic Sign][image5] ![Snow-Covered Slippery Road German Traffic Sign][image6] 
![Priority Road German Traffic Sign][image7] ![Keep Left German Traffic Sign][image8]

The model failed to classify all of these images. The second and third images from above are distorted to the point that they cannot be read even by humans so require higher quality for the model to be able to classify them. The rest of the images seem to have features that were not included in the training dataset. I say this because the model is able to predict with 100% accuracy images that are clearly visible and have limited features in the background that might change how the German traffic sign within the image look. Comparing these results to the test accuracy of 0.9585, this seems to make sense as there are images that tend to be blurry possibly to the point of the model being unable to recognize the features that fall into its corresponding classification. An example of such a blurry image from the training set is below: 

![Blurry Preprocessed Image][image9]

As a result, this model needs more images that represent real-world scenarios to train and validate on before being used to help self-drive a vehicle.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Narrows On the Right    		| Speed Limit (50 km/h)	| 

| End of All Speed and Passing Limits	| Speed Limit (60 km/h)	|

| Slippery Road				| Right-of-Way At the Next Intersection|

| Priority Road		      		| Ahead Only 		|

| Keep Left				| No Passing		|


The model was able to correctly guess 0 of these 5 traffic signs, which gives an accuracy of 0%. Testing the model on 44 test images with at least one image for each class ID gives an accuracy of 43.18%. This compares unfavorably to the accuracy on the test set of 95.85%. However, when images that are similar in quality and features as the training set, the model gives an accuracy of 100%. This makes sense as mentioned above.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the IPython Notebook below the cell with the "Output Top 5 Softmax Probabilities For Each Image Found on the Web" headline.

For almost all of the five chosen images to show for this writeup, the correct prediction does not fall in the top five softmax probabilities. We can see this in the images below:

!["Road Narrows On the Right German Traffic Sign Softmax Probabilities"][image10]
!["End of All Speed and Passing Limits German Traffic Sign Softmax Probabilities"][image11]
!["Snow-Covered Slippery Road German Traffic Sign Softmax Probabilities"][image12]
!["Priority Road German Traffic Sign Softmax Probabilities"][image13]
!["Keep Left German Traffic Sign Softmax Probabilities"][image14]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
In the first convolutional layer output, we can see that the model learns the outline shape of the German traffic sign. In the second and third layers, we see seemingly random black and white checkered patterns that could possibly be similar to utilizing bits to classify German traffic signs from images. I feel that the network is actually utilizing one or all of the layers to learn the colors that are associated with each German traffic sign classification, but appears to be random black and white checkered patterns due to the size of the features for each of the second and third layers.


