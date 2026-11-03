# Project Plan

## Project title
AI-Based Handwritten Digit Recognition System

## Problem statement
Reading handwritten digits automatically is a common image-recognition task. This project trains a CNN to classify a single handwritten digit into one of ten classes (0–9).

## Objectives
1. Load and preprocess the MNIST dataset.
2. Build and train a CNN using TensorFlow/Keras.
3. Evaluate the model on the separate MNIST test set.
4. Plot training/validation curves and a confusion matrix.
5. Provide a small interface to upload and classify a digit image.

## Proposed workflow
1. Set up Python environment and install dependencies.
2. Load the 60,000 MNIST training images and 10,000 test images.
3. Normalize image pixels to the range 0–1 and add the channel dimension.
4. Train the CNN with a validation split.
5. Evaluate on the held-out test set.
6. Save the model, plots, and metrics.
7. Run the Streamlit app and test with custom images.
8. Record actual observations and limitations in the report.

## Model outline
- Input: 28 × 28 grayscale image
- Convolution layer: 32 filters, 3 × 3, ReLU
- Max pooling: 2 × 2
- Convolution layer: 64 filters, 3 × 3, ReLU
- Max pooling: 2 × 2
- Flatten
- Dense layer: 128 units, ReLU
- Dropout: 0.3
- Output: 10 units, softmax

## Evaluation
- Test accuracy and test loss
- Confusion matrix
- Training and validation accuracy/loss curves
- Classification report by digit

## Work plan
| Stage | Work | Output |
|---|---|---|
| 1 | Environment setup | Working Python environment |
| 2 | Dataset loading and preprocessing | Normalized MNIST arrays |
| 3 | CNN design and training | Saved Keras model |
| 4 | Evaluation | Metrics and plots |
| 5 | Interface | Upload-and-predict app |
| 6 | Testing and report | Screenshots and report with real results |

## Limitations
The model is trained on MNIST, whose digits are centered and grayscale. Real photos, cluttered backgrounds, multiple digits, and very different handwriting may reduce performance. The displayed confidence is not a guarantee of correctness.

## Completion checklist
- [ ] Install requirements
- [ ] Run `python train.py`
- [ ] Confirm model and output files were created
- [ ] Run `streamlit run app.py`
- [ ] Test multiple single-digit images
- [ ] Record actual accuracy/loss and include plots
- [ ] Write the final report in your own words
