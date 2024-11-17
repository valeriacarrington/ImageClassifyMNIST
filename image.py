import numpy as np
import matplotlib.pyplot as plt

# Function to load the MNIST images from the idx3-ubyte file
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # The first four bytes represent the magic number (2051 for images)
        magic = np.frombuffer(f.read(4), dtype='>i4')[0]
        if magic != 2051:
            raise ValueError("Invalid MNIST image file format.")

        # Read the number of images
        num_images = np.frombuffer(f.read(4), dtype='>i4')[0]

        # Read the number of rows and columns in each image
        num_rows = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_cols = np.frombuffer(f.read(4), dtype='>i4')[0]

        # Read the image data and reshape into the shape (num_images, num_rows, num_cols)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

        return images

# Function to load the MNIST labels from the idx1-ubyte file
def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # The first four bytes represent the magic number (2049 for labels)
        magic = np.frombuffer(f.read(4), dtype='>i4')[0]
        if magic != 2049:
            raise ValueError("Invalid MNIST label file format.")

        # Read the number of labels
        num_labels = np.frombuffer(f.read(4), dtype='>i4')[0]

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

# Path to the archive folder containing MNIST files
folder_path = 'C:\\Users\\Valeria\\Downloads\\archive\\'

# Load the images and labels
images = load_mnist_images(folder_path + 't10k-images.idx3-ubyte')
labels = load_mnist_labels(folder_path + 't10k-labels.idx1-ubyte')

# Normalize the images to range [0, 1]
images = images.astype('float32') / 255.0

# Check the shape of the loaded images and labels
print(f"Loaded {images.shape[0]} images of size {images.shape[1]}x{images.shape[2]}")
print(f"Loaded {labels.shape[0]} labels.")

# Visualize the first image and its label (to ensure it's loaded properly)
plt.imshow(images[0], cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()
