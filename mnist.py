from data_loader import load_training_data, render_image
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

IMG_DIM = 784
MLP_HIDDEN_DIM = 32
N_CLASSES = 10
LEARNING_RATE = 0.0001

class Layer(ABC):
    @abstractmethod
    def forward(self, x_in: npt.NDArray) -> npt.NDArray:
        """
        Given x_in, return x_out
        """
        pass

    @abstractmethod
    def backward(self, g_out: npt.NDArray) -> npt.NDArray:
        """
        Given g_out (dL/dx_out) return g_in (dL/dx_in)

        Also calculate dL/dP, where P are the params of this layer.
        And apply optimizer updates for P.
        """
        pass


class ReLU(Layer):
    def forward(self, x_in: npt.NDArray) -> npt.NDArray:
        self.x_in = x_in
        x_out = x_in * (x_in > 0)
        return x_out
    
    def backward(self, g_out: npt.NDArray) -> npt.NDArray:
        # This layer needs x_in saved, i.e., activation stashing for
        # RaggedMatmul1Out
        g_in = g_out * (self.x_in.T > 0)
        return g_in

class Linear(Layer):
    def __init__(self, dim_in: int, dim_out: int):
        # TODO (anujkalia): Better init
        self.W = np.random.randn(dim_out, dim_in).astype(np.float32)

    def forward(self, x_in: npt.NDArray) -> npt.NDArray:
        self.x_in = x_in
        return self.W @ x_in
    
    def backward(self, g_out: npt.NDArray) -> npt.NDArray:
        dL_dW = g_out.T @ self.x_in.T # (d_out, 1) @ (1, d_in) = (d_out, d_in)
        ret = g_out @ self.W # (1, d_out) @ (d_out, d_in) = (1, d_in)
        self.W = self.W - (LEARNING_RATE * dL_dW)

        return ret
    

class LossFn:
    def __init__(self):
        self.ground_truth = np.array([])

    def forward(self, x_in: npt.NDArray, ground_truth: npt.NDArray) -> npt.NDArray:
        self.x_in = x_in
        self.ground_truth = ground_truth

        delta = (ground_truth - x_in)
        return -np.array([np.sum(delta ** 2)])
    
    def backward(self, g_out: npt.NDArray):
        ret = 2 * (self.x_in - self.ground_truth)
        return ret.reshape(1, N_CLASSES)


class MnistClassifier:
    def __init__(self):
        self.matmul1 = Linear(dim_in=IMG_DIM, dim_out=MLP_HIDDEN_DIM)
        self.relu1 = ReLU()

        self.matmul2 = Linear(dim_in=MLP_HIDDEN_DIM, dim_out=N_CLASSES)
        self.relu2 = ReLU()

        self.loss_fn = LossFn()

    def forward_backward(self, image: npt.NDArray, label: int):
        # One-hot ground-truth
        ground_truth = np.zeros((N_CLASSES, 1), dtype=np.float32)
        ground_truth[label, 0] = 1

        x = self.matmul1.forward(x_in=image)
        # print(f"after matmul1: {x=}")
        x = self.relu1.forward(x_in=x)
        # print(f"after relu1: {x=}")
        x = self.matmul2.forward(x_in=x)
        # print(f"after matmul2: {x=}")
        x = self.relu2.forward(x_in=x)
        # print(f"after relu2: {x=}")
        x = self.loss_fn.forward(x_in=x, ground_truth=ground_truth)
        # print(f"after loss_fn: {x=}")
        print(f"loss = {x}")

        g = self.loss_fn.backward(g_out=np.array([]))
        g = self.relu2.backward(g_out=g)
        g = self.matmul2.backward(g_out=g)
        g = self.relu1.backward(g_out=g)
        g = self.matmul1.backward(g_out=g)


if __name__ == "__main__":
    np.random.seed(42)

    images, labels = load_training_data()
    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"images[0] shape: {images[0].shape}")
    print(f"labels[0]: {labels[0]}")

    c = MnistClassifier()
    n_images = images.size
    for i in range(0, 400):
        c.forward_backward(image=images[i], label=labels[i])