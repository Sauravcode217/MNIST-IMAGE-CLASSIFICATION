import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scale it between 0 and 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to include the channel dimension (since CNNs expect 3D data)
# The MNIST images are grayscale (1 channel), so we add the channel dimension.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the CNN model
model = tf.keras.models.Sequential([
    # First Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Max Pooling layer with a 2x2 window
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Another Max Pooling layer
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten the output from the convolutional layers to feed into fully connected layers
    tf.keras.layers.Flatten(),
    
    # Fully connected (dense) layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.5),
    
    # Output layer with 10 neurons (one for each digit) and softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on test data
model.evaluate(x_test, y_test)
