# Dataset Link
https://data.mendeley.com/datasets/rscbjbr9sj/2

# Pneumonia Diagnosis using CNN with Transfer Learning

This project presents a Convolutional Neural Network (CNN) model that utilizes Transfer Learning for Pneumonia diagnosis through chest X-ray images. Our group developed a model to classify images into two categories: `NORMAL` and `PNEUMONIA`.

## Dataset Details

- **Source:** Kaggle’s Chest X-ray Images (Pneumonia) dataset.
- **Total Images:** 5216  
  - `NORMAL`: 1341 images  
  - `PNEUMONIA`: 3875 images
- **Data Split:** Training (70%), Validation (15%), Testing (15%)
- **Class Imbalance:** The dataset is imbalanced, with a higher number of Pneumonia cases. To address this, we applied data augmentation techniques to the training set.

## Preprocessing & Data Augmentation

- **Image Rescaling:** Pixel values were scaled to the range `[0, 1]` using `ImageDataGenerator(rescale=1/255)`.
- **Image Resizing:** All images were resized to `(224, 224)` to match the input shape required by the model.
- **Data Augmentation Techniques:**
  - Rotation Range: `20 degrees`
  - Width Shift Range: `0.2`
  - Height Shift Range: `0.2`
  - Horizontal Flip
  - Zoom Range: `0.2`

## Model Architecture

We implemented **Transfer Learning using ResNet50**, a robust and efficient architecture for image classification tasks. The following modifications were made to the base model:

- Removed the top layer (`include_top=False`)
- Added a `GlobalAveragePooling2D()` layer
- Added a `Dense(512)` layer with `ReLU` activation and a `Dropout` layer for regularization
- Included a final `Dense(1)` layer with `Sigmoid` activation for binary classification

### Why ResNet50?
- Its residual learning framework helps avoid the vanishing gradient problem, enabling effective training of deeper networks.
- Pretrained weights from `ImageNet` accelerate training and improve performance by leveraging learned features from a large-scale dataset.

## Training & Hyperparameters

- **Optimizer:** Adam (learning rate: `0.001`)
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** `32`
- **Epochs:** `10`
- **Callbacks:** EarlyStopping was used to monitor validation loss and reduce the risk of overfitting.

## Performance Metrics

- **Accuracy:** `92.80%`
- **Precision, Recall, F1-score:** Not explicitly calculated. Including these metrics would provide a more well-rounded evaluation of the model’s performance.

## Limitations & Improvements

### Limitations
- Our model is slightly biased towards the `PNEUMONIA` class due to class imbalance.
- A limited number of epochs may have restricted the model’s ability to fully converge.
- Sole reliance on accuracy may not be sufficient; additional metrics like precision, recall, and F1-score are necessary for deeper evaluation.

### Improvements
- Implementing `class_weight` or using techniques like `SMOTE` to better handle class imbalance.
- Training for more epochs with EarlyStopping enabled for adaptive learning.
- Incorporating precision, recall, and F1-score for a more comprehensive performance report.
- Performing hyperparameter tuning to further optimize the model.
- Exploring alternative architectures such as `InceptionV3` or `EfficientNet`.
- Applying cross-validation for more reliable model validation.

## Conclusion

Our project demonstrates the effectiveness of **Transfer Learning with ResNet50** in detecting Pneumonia from chest X-ray images, achieving a promising accuracy of `92.80%`. While the results are encouraging, improvements in evaluation metrics and handling of class imbalance would further enhance the model's reliability and real-world applicability.
