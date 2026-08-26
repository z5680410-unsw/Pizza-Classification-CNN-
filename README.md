# Pizza Classification (CNN)

Classifying pizza images as plain cheese or pepperoni using a Convolutional Neural Network built with Keras/TensorFlow, trained on the Pizza10 image dataset.

## Problem

Automated visual quality checks (e.g. "was the correct pizza made?") can reduce manual inspection and catch order errors before they reach the customer. This project builds a simple binary image classifier to distinguish plain cheese pizza from pepperoni pizza.

## Data

600 training images (300 Cheese, 300 Pepperoni) and 200 testing images (100 Cheese, 100 Pepperoni), each 100x100 RGB, organised into `Cheese/` and `Pepperoni/` subfolders. Subset of the annotated multi-ingredient [Pizza10](https://arxiv.org/abs/2012.02821) dataset.

## Approach

- Loaded and preprocessed images (BGR→RGB conversion, resized to 100x100, normalised to [0,1])
- Built a Sequential CNN: 3x [Conv2D + MaxPooling2D] blocks (32, 32, 64 filters), Dropout (0.4), Flatten, Dense(128, ReLU), Dense(2, Softmax)
- Compiled with Adam optimiser and sparse categorical crossentropy loss
- Trained for 10 epochs with validation on the test set
- Evaluated using a confusion matrix and classification report

## Results

**Test set accuracy: 93%**

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Cheese | 0.93 | 0.93 | 0.93 |
| Pepperoni | 0.93 | 0.93 | 0.93 |

Confusion matrix:

|              | Predicted Cheese | Predicted Pepperoni |
|---|---|---|
| **Actual Cheese**    | 93 | 7 |
| **Actual Pepperoni** | 7  | 93 |

Validation accuracy improved steadily from 74.5% (epoch 1) to 93% (epoch 10), tracking training accuracy closely with no significant overfitting.

## Key Insight

A relatively shallow 3-layer CNN reaches 93% accuracy on this task within just 10 epochs, showing that visually distinct classes (plain vs. topped pizza) don't require deep architectures to separate reliably.

## Tools & Libraries

Python, TensorFlow/Keras, OpenCV, scikit-learn, matplotlib, seaborn

## Notebook

Open the `.ipynb` file directly on GitHub to view the full code, outputs, and visualisations. `Training.zip` and `Testing.zip` contain the image data used.
