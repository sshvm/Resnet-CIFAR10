# ResNet CIFAR-10 Image Classifier

A TensorFlow/Keras image-classification project that trains a ResNet-style convolutional neural network on the CIFAR-10 dataset. The project is implemented in a Jupyter notebook and includes a saved Keras model, training plots, a confusion matrix, and a model summary.

## Results

The notebook trained for 30 epochs with a batch size of 128. Its final epoch reported 94.42% training accuracy and 86.33% accuracy when evaluated on the CIFAR-10 test split.

**Evaluation note:** the notebook also passes this test split as `validation_data` during training. The 86.33% score is therefore not an independent final test result. For a clean evaluation, split the original training data into training and validation sets, use validation data during training, and evaluate the test set only after model choices are complete.

## Model

The model accepts RGB images of shape `32 × 32 × 3` and predicts one of ten classes. It includes:

- Random horizontal flips, rotations, and zoom for data augmentation
- Convolutional layers with 64, 128, and 256 filters
- Residual blocks with skip connections and batch normalization
- Max pooling, global average pooling, and dropout
- A 10-class softmax output layer

The saved model contains approximately 2.3 million trainable parameters.

## Dataset

CIFAR-10 contains 60,000 color images sized 32 × 32 pixels: 50,000 training images and 10,000 test images. Keras downloads the dataset automatically when the notebook runs. The class labels are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Getting started

Use Python 3 with pip. From the project directory, install the dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
jupyter notebook notebook.ipynb
```

Run the notebook cells from top to bottom to load CIFAR-10, build and train the model, evaluate it, and create the plots. Training can take a while; a GPU is helpful but not required. The first notebook cell also contains a hosted-notebook `pip install` command, which can be skipped when dependencies are already installed locally.

## Use the saved model

The repository includes `resnet_cifar10.keras`. The model expects image values scaled to the range `[0, 1]`:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

model = tf.keras.models.load_model("resnet_cifar10.keras")

(_, _), (x_test, _) = cifar10.load_data()
image = x_test[0].astype("float32") / 255.0
probabilities = model.predict(np.expand_dims(image, axis=0), verbose=0)
predicted_class = int(np.argmax(probabilities, axis=1)[0])
print(predicted_class)
```

The class indices are ordered as: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Repository contents

```text
.
├── notebook.ipynb
├── resnet_cifar10.keras
├── requirements.txt
├── Summary/
│   └── model_summary.txt
└── screenshots/
    ├── confusion_matrix.png
    ├── training Vs Val accuracy.png
    └── val_accuracy_value.png
```

## Visualizations

![Training and validation accuracy](<screenshots/training Vs Val accuracy.png>)

![Confusion matrix](screenshots/confusion_matrix.png)

![Validation accuracy](screenshots/val_accuracy_value.png)

## Dependencies

Dependencies are listed in [requirements.txt](requirements.txt): TensorFlow, NumPy, Matplotlib, scikit-learn, and Seaborn.
