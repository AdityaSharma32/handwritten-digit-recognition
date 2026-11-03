# Major Project Report
## AI-Based Handwritten Digit Recognition System

**Student name:** [Write your name]  
**Course / semester:** [Fill in]  
**Institution:** [Fill in]  
**Date:** [Fill in]

## 1. Introduction
Explain the handwritten digit recognition problem and why image classification is useful. Write this section in your own words.

## 2. Objectives
- Train a CNN to recognize digits from 0 to 9.
- Use the MNIST dataset for training and testing.
- Evaluate the model using accuracy, loss, and a confusion matrix.
- Build an interface for classifying an uploaded digit image.

## 3. Dataset
The project uses MNIST through TensorFlow/Keras. The standard split contains 60,000 training images and 10,000 test images. Each image is a 28 × 28 grayscale image with a label from 0 to 9.

## 4. Preprocessing
Describe normalization from 0–255 to 0–1, adding the channel dimension for CNN input, and the preprocessing applied to uploaded images.

## 5. Model architecture
Describe the two convolution/max-pooling stages, flattening, dense layer, dropout, and 10-class softmax output. Add a screenshot of the model summary if required.

## 6. Training setup
- Epochs: 5
- Batch size: 128
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Validation split: 10% of the training data

Record your computer details and any changes you made to the settings.

## 7. Results
Fill this section only after running `python train.py`.

- Test accuracy: [copy from outputs/metrics.json]
- Test loss: [copy from outputs/metrics.json]
- Training observations: [describe your actual plots]
- Confusion matrix observations: [mention digits that were often confused, if any]

Insert `outputs/training_history.png` and `outputs/confusion_matrix.png`.

## 8. Application
Add screenshots of the Streamlit app and examples of uploaded images with predictions. Explain how to run it.

## 9. Limitations
Discuss how results can be affected by image quality, centering, handwriting style, and the difference between MNIST images and real photographs.

## 10. Conclusion and future improvements
Write a conclusion based on your actual evaluation. Possible future improvements include a drawing canvas, image-centering improvements, data augmentation, and comparison with an MLP.

## 11. References
- MNIST database: https://yann.lecun.com/exdb/mnist/
- TensorFlow Keras datasets: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
- TensorFlow documentation: https://www.tensorflow.org/
- Streamlit documentation: https://docs.streamlit.io/

## Viva preparation
Prepare to explain:
1. What is a CNN?
2. Why are pixel values normalized?
3. What does a convolution layer learn?
4. Why is max pooling used?
5. What does softmax output?
6. What is the difference between training and test data?
7. What does a confusion matrix show?
8. Why might the model misclassify a real-world image?
