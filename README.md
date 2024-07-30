# Fruit Picking System Using Computer Vision

## Overview

This project demonstrates a computer vision-based robotic fruit-picking system developed at Vellore Institute of Technology, Bhopal. It leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to automate the process of fruit harvesting. The system integrates advanced robotics and computer vision to identify and pick ripe fruits with precision.

## Key Components

- **Deep Learning Model**: Utilizes CNNs to classify and detect various types of fruits from images. The model is trained on a diverse dataset of fruit images, employing techniques such as data augmentation and normalization to enhance performance.

- **Robotic Arm**: Guided by the CNN predictions, the robotic arm is designed to perform accurate fruit-picking maneuvers based on the fruit's location and ripeness.

- **Data Processing**: Involves preprocessing of fruit images, including resizing, normalization, and augmentation to prepare a robust dataset for training and testing.

## Methodology

1. **Data Preprocessing**: Collection and cleaning of fruit images, resizing to uniform resolution (e.g., 100x100 pixels), and augmentation to reflect real-world variability.

    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    ```
  ![Block Diagram](https://github.com/srijaaaaa/Automated-Fruit-Recognition-and-Picking-System/blob/main/1.2.jpg)
2. **Model Architecture**: Built using Keras with layers for feature extraction (`Conv2D`), down-sampling (`MaxPooling2D`), and dropout for regularization.

    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # num_classes: number of fruit categories
    ])
    ```

3. **Model Training**: Compiled with categorical cross-entropy loss and optimized using Stochastic Gradient Descent (SGD). Trained over 50 epochs with metrics such as accuracy and loss tracked.

    ```python
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=50
    )
    ```

4. **Evaluation & Enhancement**: Model performance evaluated on test data. For example, if the test accuracy is 85% and loss is 0.4, enhancements based on these metrics are applied to improve generalization and robustness.

    ```python
    loss, accuracy = model.evaluate(test_generator, steps=50)
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Loss: {loss:.2f}")
    ```

## Results

The system successfully classified and picked fruits, demonstrating effective learning and generalization from the training data. Performance metrics showed improvements, with accuracy increasing from an initial 70% to 85% and loss reducing from 0.6 to 0.4 over the training period.

## Future Work

Further development will focus on enhancing the system's capability to handle diverse orchard environments and improving the robotic arm's efficiency.

## Acknowledgements

Special thanks to Vellore Institute of Technology, Bhopal, and mentor Jayanthi Mam for their support and guidance throughout this project.
