Image Classification using CNN

 This repository contains an implementation of Convolutional Neural Networks (CNNs) for image classification, focusing on the CIFAR-10 dataset. The project demonstrates how CNNs can be used to classify images into various categories efficiently, showcasing the versatility of deep learning techniques.

Project Overview

 The project leverages CNNs to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 RGB images across 10 categories such as airplanes, cars, birds, and dogs. It includes:
 Preprocessing and normalizing the dataset.
 Designing a CNN architecture with convolutional layers, pooling layers, and fully connected layers.
 Training the model and optimizing parameters for high accuracy.
 Evaluating model performance using metrics like accuracy, precision, and recall.

Features

 Custom CNN Architecture: Built specifically for the CIFAR-10 dataset to optimize feature extraction and classification.
 Data Augmentation: Techniques like random rotations and flips are applied to increase dataset variability.
 Model Evaluation: Detailed analysis using metrics such as confusion matrix, precision, recall, and F1-score.
 Visualization: Includes visual representation of training progress, validation accuracy, and loss curves.

Incase of any errors

 The error Unrecognized keyword arguments: ['batch_shape'] indicates that the model file cifar10_model.h5 was saved in a way that is not fully compatible with your current TensorFlow/Keras version. To resolve this, we need to ensure the model is loaded and re-saved in a compatible format.

Contributions

 Contributions are welcome! Fork this repository, make your changes, and submit a pull request.
