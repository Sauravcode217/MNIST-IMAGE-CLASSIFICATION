# MNIST-IMAGE-CLASSIFICATION
# Implemented a Convolutional Neural Network (CNN) for Image Classification using TensorFlow and Keras:

Developed a CNN to classify handwritten digits from the MNIST dataset, which contains 70,000 grayscale images of digits (0-9).

Preprocessed the dataset by normalizing pixel values and reshaping images to fit the input format required by CNNs.

Built a sequential CNN model architecture consisting of:
Two convolutional layers (with 32 and 64 filters, respectively) for feature extraction using 3x3 kernels and ReLU activation.
Max-pooling layers to downsample feature maps, improving computational efficiency.
Fully connected (Dense) layer with 128 neurons, followed by Dropout to prevent overfitting.
An output layer with 10 neurons (softmax activation) for multi-class classification of digits.

Compiled the model using the Adam optimizer and sparse categorical crossentropy loss function, monitoring accuracy during training.

Achieved high classification accuracy (around 98-99%) on the test dataset after training for 5 epochs.
