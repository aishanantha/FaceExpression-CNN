FaceExpression-CNN
# Project Title: Facial Emotion Detection using CNN and OpenCV
The goal of facial emotion detection, a crucial application of deep learning and computer vision, is to automatically identify human emotions from facial expressions. This project uses OpenCV for face detection and real-time implementation, and Convolutional Neural Networks (CNN) for learning expression features. To classify emotions like Happy, Sad, Angry, Fear, Surprise, Neutral, and Disgust, the model is trained on a labeled facial expression dataset (FER-2013 from Kaggle). Real-time, reliable, and scalable emotion recognition is made possible by the combination of OpenCV's effective image processing and CNN's spatial learning capabilities.
## Key Features and Components:
#### OpenCV -Used for image processing, face detection, and video capture (real-time webcam feed).
#### CNN (Keras/TensorFlow)- Deep learning model trained to classify facial expressions.
#### Dataset- FER-2013 dataset; grayscale 48x48 images with 7 emotion labels.
#### Preprocessing -	Face detection using Haar cascades, resizing, grayscale conversion, normalization.
#### Training - CNN architecture with multiple convolution and pooling layers, dropout, and softmax output layer.
#### Real-Time Inference	Captures - video via webcam, detects faces using OpenCV, predicts emotion using trained model.
#### Evaluation - Accuracy, loss graphs, confusion matrix, and model performance on validation/test data.
## Flow Chart of the process:
![pipeline](https://github.com/user-attachments/assets/4b8abc0f-7616-43a4-82f1-dd48d40f21b0)



