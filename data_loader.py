from pathlib import Path
import struct
from typing import Tuple

import numpy as np


DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
TRAIN_IMAGES_FILE = DATASETS_DIR / "train-images-idx3-ubyte"
TRAIN_LABELS_FILE = DATASETS_DIR / "train-labels-idx1-ubyte"
TEST_IMAGES_FILE = DATASETS_DIR / "t10k-images-idx3-ubyte"
TEST_LABELS_FILE = DATASETS_DIR / "t10k-labels-idx1-ubyte"
ASCII_GRADIENT = " .:-=+*#%@"


def _read_mnist_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number in {path}: {magic}")

        image_data = np.frombuffer(f.read(), dtype=np.uint8)

    expected_size = count * rows * cols
    if image_data.size != expected_size:
        raise ValueError(
            f"Image file size mismatch in {path}: expected {expected_size} bytes, got {image_data.size}"
        )

    return image_data.reshape(count, rows * cols, 1)


def _read_mnist_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number in {path}: {magic}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    if labels.size != count:
        raise ValueError(
            f"Label file size mismatch in {path}: expected {count} labels, got {labels.size}"
        )

    return labels


def _load_dataset(images_file: Path, labels_file: Path, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if not DATASETS_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing dataset directory: {DATASETS_DIR}. "
            "Place the MNIST data files in datasets/."
        )

    if not images_file.is_file():
        raise FileNotFoundError(f"Missing {dataset_name} images file: {images_file}")

    if not labels_file.is_file():
        raise FileNotFoundError(f"Missing {dataset_name} labels file: {labels_file}")

    images = _read_mnist_images(images_file)
    labels = _read_mnist_labels(labels_file)

    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{dataset_name.capitalize()} image count does not match label count: "
            f"{images.shape[0]} != {labels.shape[0]}"
        )

    return images, labels


def load_training_data() -> Tuple[np.ndarray, np.ndarray]:
    return _load_dataset(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE, "training")


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    return _load_dataset(TEST_IMAGES_FILE, TEST_LABELS_FILE, "test")


def render_image(image: np.ndarray, rows: int = 28, cols: int = 28) -> str:
    flat_image = np.asarray(image, dtype=np.uint8).reshape(rows * cols)
    pixels = flat_image.reshape(rows, cols)
    gradient = np.array(list(ASCII_GRADIENT))

    indices = (pixels.astype(np.uint16) * (len(gradient) - 1)) // 255
    ascii_rows = ["".join(gradient[row]) for row in indices]
    rendered = "\n".join(ascii_rows)
    print(rendered)
    return rendered
