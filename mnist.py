from data_loader import load_training_data, render_image
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class Layer(ABC):
    @abstractmethod
    def forward(self, x_in: np.array) -> np.array:
        """
        Given x_in, return x_out
        """
        pass

    def backward(self, g_out: np.array, x_in: Optional[np.array]) -> np.array:
        """
        Given g_out (dL/dx2) return g_in (dL/dx1).

        Also calculate dL/dP, where P are the params of this layer.
        And apply optimizer updates for P.
        """
        pass


class ReLU(Layer):
    def forward(self, x_in: np.array) -> np.array:
        x_out = x_in * (x_in > 0)
        return x_out
    
    def backward(self, g_out: np.array, x_in: np.array) -> np.array:
        g_in = g_out * (x_in > 0)
        return g_in


if __name__ == "__main__":
    images, labels = load_training_data()
    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"images[0] shape: {images[0].shape}")
    print(f"labels[0]: {labels[0]}")
    render_image(images[3])
    print(labels[3])

    x: Layer = ReLU()