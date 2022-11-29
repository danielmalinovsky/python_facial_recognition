# Face Recognition & Detection in Python
Repository for the _Facial Image Recognition_ project in Python within the course _**"Agile Project of Machine Learning Applications"**_ **(4IZ563)** at Faculty of Informatics and Statistics, Prague University of Economics and Business (**VŠE**).

_**Authors**_: [**Daniel Malinovsky**](https://www.linkedin.com/in/daniel-malinovsky-88b162198) (_Team Leader_), [**Petr Nguyen**](https://www.linkedin.com/in/petr-ngn) (_Lead Data Scientist_), [Petr Hollmann](https://www.linkedin.com/in/petr-hollmann-3583aa208), [Roman Pavlata](https://www.linkedin.com/in/roman-pavlata-a3b602161), [Natalie Musilova](https://www.linkedin.com/in/natálie-musilová-3b98287a) (_regular members of the_ _**Data Science & Machine Learning team**_)

This course is being surpervised by the Data Scientists and AI & ML Engineers at [**CN Group CZ**](https://www.linkedin.com/company/cngroup-dk), namely by [Petr Polak](https://www.linkedin.com/in/87petrpolak), [Patrik Tison](https://www.linkedin.com/in/patriktison), Viktor Stepanuk, Marek Hronek and [Tomas Kliment](https://www.linkedin.com/in/tomáš-kliment-b74120196).



_**Deliverable**_: Face Recognition & Detection tool implemented in Python
- Business and Organizational tasks (_Backlogs and estimates_)
- Agile Development in Machine Learning (_using packages such_ `Tensorflow` _or_ `Keras`)
- Other tasks (_Presentations of sprints, regular communication with the Product Owners, SCRUM practices_)


## `facial_recognition_v0.1`
- Loading sample data
- Cropping out bounding boxes
- Splitting the data
- Implementing Bounding boxes generator
- Distribution of images
- Exploratory Data Analysis (_EDA_) of images' attributes
- Implementing balanced-pairs generator
- Classification with Residual Neural Network (_ResNet_)

## Sprint 1 
#### Deadline: 18/10/2022
_**Tasks**_:
1. _**[DONE]**_ Prepare enviroment for running code
2. _**[DONE]**_ Find dataset used for training a model
   - please see: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

3. _**[DONE]**_ Load dataset and ETL (_if needed_)
   - please see: `facial_recognition_v0.1`

## Sprint 2
#### Deadline: 08/11/2022
_**Tasks**_:
1. _**[DONE]**_ Train/Validation/Test split (_60%/20%/20%_)
2. _**[DONE]**_ Bound box creation & image cropping
3. _**[DONE]**_ Resizing of cropped images

## Sprint 3
#### Deadline: 15/11/2022
_**Tasks**_:
1. _**[DONE]**_ Distribution of photos per person
2. _**[DONE]**_ EDA of attributes (_histograms, correlation analysis_)
3. _**[DONE]**_ Generator of balanced random pairs (_pairs of 2 photos of the same person/pairs of 2 photos of 2 different persons_)
4. _**[DONE]**_ ResNet 50 with Keras application
5. _**[DONE]**_ Binary Classification using ResNet on multiple inputs

## Sprint 4
#### Deadline: 29/11/2022
_**Tasks**_:
1. _**[Will be done within the next sprint]**_ Create data processing pipeline for processing face images (_using Tensorflow data API_)
2. _**[DONE]**_ Training process evaluation (_plotting loss functions, confusion matrices etc._)
3. _**[DONE]**_ Classification of each team member's photo (_with uploading photos and subsequent cropping using a face detector and predicting feature vectors_)

## Sprint 5
#### Deadline: 13/12/2022
1. Generate triplets
   - _Triplet contains 2 different photos of the same person and 1 photo of somebody else_
   - _Create a function that returns a given number of triplets_
   - _Generate triplets for NN training_
2. Create model using contrastive loss
   - _Assemble model with pairs of images on input_
   - _Use pre-trained image classification architecture as a feature extractor_
   - _Using contrastive loss as a cost function_
   - _Use feature extractor from model to make predictions_
   - _Evaluate predictions using some distance measure (for example Euclidean distance)_
   - _Estimate threshold for the distance between positive/positive and negative/negative_
   - _Predict on pair of images using distance measure and selected threshold_
3. Create model using triplet loss
   - _Assemble model with triplets with images on input_
   - _Use pre-trained image classification architecture as a feature extractor_
   - _Using triplet loss as a cost function_
   - _Use feature extractor from model to make predictions_
   - _Evaluate predictions using some distance measure (for example Euclidean distance)_
   - _Estimate threshold for the distance between positive/positive and negative/negative_
   - _Predict on pair of images using distance measure and selected threshold_
4. Train model on Google Colab
   - _Pick the model you want to try first from the three architectures_
   - _Copy images of persons for training to Google Drive_
   - _Mount Google Drive in Colab_
   - _Train the model on the images_
   - _Trained model on at least 5000 samples of images (pairs or triplets)_
   - _Save the trained model and loss function progress during training_
   - _Download saved model to local machine_
   - _Run inference on local machine with sample of images (from a testing dataset)_

## Final Sprint
#### Deadline: January 2023
_TBA_
