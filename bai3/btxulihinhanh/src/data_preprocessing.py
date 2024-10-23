import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image, UnidentifiedImageError

def load_and_preprocess_image(image_path, img_size=(64, 64)):
    """
    Load an image from the specified path and preprocess it by resizing and normalizing.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(img_size)
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return image_array.flatten()  # Flatten the image to a 1D array for classical ML algorithms
    except UnidentifiedImageError:
        print(f"Warning: {image_path} is not a valid image file. Skipping...")
        return None

def load_dataset(dataset_dir, img_size=(64, 64)):
    """
    Load dataset, resize and flatten the images, and return as X (features) and y (labels).
    The 'dataset_dir' should contain two subfolders: 'animals' and 'flowers'.
    """
    X = []
    y = []
    class_map = {'flowers': 0, 'animals': 1}  # 0 for flowers, 1 for animals
    
    for class_name, label in class_map.items():
        class_dir = Path(dataset_dir) / class_name
        for image_path in class_dir.iterdir():
            if image_path.suffix in ['.jpg', '.jpeg', '.png']:
                image = load_and_preprocess_image(image_path, img_size)
                if image is not None:  # Only add valid images
                    X.append(image)
                    y.append(label)
    return np.array(X), np.array(y)

def split_data(X, y, test_size=0.2):
    """
    Split dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)
