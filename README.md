# Cat vs Dog Image Classification using CNN

This project is a simple image classification project where I built a **Convolutional Neural Network (CNN)** to classify images as either a **cat** or a **dog**.

I created this project using **Python, TensorFlow, and Keras** in Google Colab. I first trained a basic CNN model, then tried to improve its performance using **data augmentation** and **dropout**.

The basic model gave around **74% accuracy**, while the improved model reached around **79% accuracy**.

---

## Tools and Libraries Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

## Dataset

The dataset contains separate images of cats and dogs for training and testing.

The images are resized to:

```text id="x6ipmx"
150 × 150
```

Since the images are RGB images, the input shape used in the model is:

```python id="ebmvmn"
(150, 150, 3)
```

The two classes are:

```text id="yzwd2j"
Cat → 0
Dog → 1
```

---

## Importing Libraries

At the beginning, I imported the libraries needed for the project.

```python id="z0ya0g"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
```

TensorFlow and Keras are used to build and train the model, NumPy is used for array operations, and Matplotlib is used for plotting images and graphs.

---

## Connecting Google Drive

Since my dataset was stored in Google Drive, I mounted the drive in Colab.

```python id="fvd84t"
from google.colab import drive
drive.mount('/content/drive')
```

After that, I checked the dataset folder to make sure the ZIP file was available.

```python id="n9h1vn"
print(os.listdir('/content/drive/MyDrive/Datas'))
```

---

## Extracting the Dataset

The dataset was stored in a ZIP file, so I extracted it before loading the images.

```python id="e5fd3a"
import zipfile

zip_path = '/content/drive/MyDrive/Datas/cat_dog_dataset.zip'
extract_path = '/content/cat_dog_dataset'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")
```

---

## Loading the Images

I used `image_dataset_from_directory()` to load the training and testing images.

```python id="s3ojjd"
from tensorflow.keras.utils import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    '/content/cat_dog_dataset/training_set/training_set',
    image_size=(150, 150),
    batch_size=32
)

test_ds = image_dataset_from_directory(
    '/content/cat_dog_dataset/test_set/test_set',
    image_size=(150, 150),
    batch_size=32
)
```

Here, every image is resized to `150 × 150`, and the model processes 32 images at a time.

---

# Basic CNN Model

I first created a simple CNN model to get a basic result.

```python id="fzxnln"
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

The model uses three convolutional layers with 32, 64, and 128 filters.

The convolution layers help the model learn important features from the images, such as edges, shapes, textures, and other patterns.

`MaxPooling2D()` is used after each convolution layer to reduce the size of the feature maps.

The `Flatten()` layer converts the extracted features into a single vector, which is then passed to the Dense layers.

Since this is a binary classification problem, the final layer uses:

```python id="8fma7z"
layers.Dense(1, activation='sigmoid')
```

The sigmoid output gives a value between 0 and 1.

In this project:

```text id="roa3mg"
Value > 0.5 → Dog
Value ≤ 0.5 → Cat
```

---

## Compiling the Model

Before training, I compiled the model using Adam optimizer and binary crossentropy.

```python id="0t47ac"
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

I used `binary_crossentropy` because there are only two classes.

---

## Model Training

The model was trained for 10 epochs.

```python id="gjmjwa"
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds
)
```

Model training is the process where the CNN learns from the images.

During training, the model makes predictions, compares them with the actual labels, calculates the error, and updates its weights so that future predictions become better.

One epoch means the model has gone through the full training dataset once.

Since I used:

```python id="ogt0pi"
epochs=10
```

the model went through the training data 10 times.

---

## Basic Model Result

After training, I evaluated the model using the test dataset.

```python id="jwf6mb"
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)
```

The basic CNN achieved around:

```text id="zpqi37"
74.15% test accuracy
```

This was a decent starting result, but I wanted to improve the model further.

---

# Improving the Model

To improve the model, I added **data augmentation** and **dropout**.

---

## Data Augmentation

Data augmentation creates slightly different versions of the training images.

```python id="gqb4wd"
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

For example, an image may be flipped, slightly rotated, or zoomed.

This gives the model more variety during training and helps it generalize better to new images.

---

## Improved CNN Model

The improved model uses the same main CNN structure, but I added data augmentation and dropout.

```python id="3n2pn5"
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])
```

---

## Why Dropout?

I used:

```python id="xj9srq"
layers.Dropout(0.5)
```

Dropout randomly turns off some neurons during training.

This helps reduce overfitting and prevents the model from depending too much on specific neurons.

---

## Training the Improved Model

The improved model was again trained for 10 epochs.

```python id="bs0hcs"
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds
)
```

After training, I evaluated it using the test dataset.

```python id="jgh3hr"
test_loss, test_accuracy = model.evaluate(test_ds)

print("Test Accuracy:", test_accuracy)
```

The improved model achieved around:

```text id="ild6f2"
79.49% test accuracy
```

So the improved model performed better than the basic model.

---

## Model Comparison

| Model        | Test Accuracy |
| ------------ | ------------: |
| Basic CNN    |       ~74.15% |
| Improved CNN |       ~79.49% |

Adding data augmentation and dropout improved the test accuracy by around 5%.

---

# Accuracy and Loss Graphs

I also plotted the training and validation accuracy.

```python id="2gj4l5"
plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```

This graph shows how the accuracy changes over each epoch.

I also plotted the training and validation loss.

```python id="u214xk"
plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

The loss graph helps show whether the model is learning properly during training.

---

# Confusion Matrix

I used a confusion matrix to see how many cats and dogs were predicted correctly or incorrectly.

```python id="n58vbf"
class_names = ['Cat', 'Dog']

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.show()
```

The confusion matrix gives a clearer idea of where the model is making mistakes.

---

# Classification Report

I also used a classification report.

```python id="l45nk0"
from sklearn.metrics import classification_report

print(classification_report(
    y_true,
    y_pred,
    target_names=["Cat", "Dog"]
))
```

The classification report gives values such as:

* Precision
* Recall
* F1-score
* Support

These values give more information about the model than accuracy alone.

---

# Testing a New Image

Finally, I tested the trained model using new cat and dog images.

For example:

```python id="kc5kdd"
img_path = '/content/Test/dog.png'
```

Then I loaded the image and resized it to the same size used during training.

```python id="qrwgjr"
from tensorflow.keras.utils import load_img, img_to_array

img = load_img(img_path, target_size=(150, 150))

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("Prediction probability:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Predicted: Dog")
else:
    print("Predicted: Cat")
```

The same process can be used for any new cat or dog image.

---

# Saving the Model

After training, I saved the improved model so that it can be used later without training it again.

```python id="5ev81x"
model.save('/content/drive/MyDrive/Datas/cat_dog_model_improved.keras')
```

---

# Project Workflow

```text id="fdxukq"
Dataset
   ↓
Extract Dataset
   ↓
Load Cat and Dog Images
   ↓
Resize Images
   ↓
Build CNN Model
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Add Data Augmentation and Dropout
   ↓
Train Improved Model
   ↓
Check Accuracy and Loss
   ↓
Confusion Matrix
   ↓
Classification Report
   ↓
Test New Images
```

---

# Result

The final improved CNN model achieved approximately **79.49% test accuracy**.

The basic CNN worked reasonably well, but adding data augmentation and dropout helped improve the model's performance and reduced overfitting.

---

# What I Learned

Through this project, I learned the basic workflow of building an image classification model using CNN.

I learned how to:

* Prepare and load an image dataset
* Build a CNN using TensorFlow and Keras
* Train and evaluate a model
* Use convolution and pooling layers
* Apply data augmentation
* Use dropout to reduce overfitting
* Visualize training accuracy and loss
* Use a confusion matrix and classification report
* Test the trained model on new images

This project gave me a better understanding of how CNNs can be used for real-world image classification problems.
