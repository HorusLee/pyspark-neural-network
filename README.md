## Describe here your project

1. Dataset

The dataset is the same as the assignment 4 and 5. 
The small dataset (37.5 MB of text) is used for training and testing of the 
model locally. The large training dataset (1.9 GB of text) is used for training 
the model in the cloud, the large test dataset (200 MB of text) is for test the 
model in the cloud. The urls of these dataset are listed as follows.
Dataset	                                    Google Cloud Storage
Small Training Dataset (37.5 MB of Text)	gs://metcs777/SmallTrainingData.txt
Large Training Dataset (1.9 GB of Text)	    gs://metcs777/TrainingData.txt
Large Text Dataset (200 MB of Text)	        gs://metcs777/TestingData.txt

2. Model

A 2-layer fully connected neural network model has been implemented based on 
spark. The input unit is 10,000, which is equal to the dimension of the feature; 
the hidden unit is adjustable and the output unit is set to 1 because this is a 
binary classification problem. Finally, this neural network classifier can 
automatically figure out whether a text document is an Australian court case. 
To run the model, just changing the directory of the dataset and setup the 
hidden unit parameter.


# How to run  

Changing the directory of the dataset and setup the hidden unit parameter.
