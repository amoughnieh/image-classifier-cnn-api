# Image Classifier

In this project I built a ML model to classify images of cars 🚗, planes ✈️, and ships 🚤. I used scraped images from the web to train and test the model, containerized the model and deployed it to Google Cloud Platform.
I built a simple web app to communicate with the model and deployed it to Firebase.

The main goal of this project is to deploy a fully functional predictive model to the cloud, rather than focusing on optimizing its performance. However, future updates will be made to focus on optimizing the model itself.

The model was trained and tested on a dataset of 885 images, which is a very small number for any CNN model to perform well, but this was more of a Proof of Concept. The final f1-score on the test set is 78.8%.

Key features:

- Convolutional Neural Network (CNN)
- Bayesian optimization for tuning learning rate, weight decay, and epochs
- 5-fold cross-validation
- Experiment tracking with MLflow
- Model deployment using FastAPI
- Dockerized API and deployed to GCP
- Web app with a simple interface connected to the cloud model

Web app can be found here: https://image-classifier-dc8ad.web.app/
