from data_loader import load_training_data, load_test_data
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Tuple


IMG_DIM = 784
MLP_HIDDEN_DIM = 32
N_CLASSES = 10
LEARNING_RATE = 0.01


class Layer(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, x_in: npt.NDArray) -> npt.NDArray:
        """
        Given x_in, return x_out
        """
        pass

    def backward(self, g_out: npt.NDArray) -> npt.NDArray:
        """
        Shared logic that must always run first.
        """
        # print(f"g_out for {self.name} = {g_out}")
        return self._backward(g_out)

    @abstractmethod
    def _backward(self, g_out: npt.NDArray) -> npt.NDArray:
        """
        Given g_out (dL/dx_out) return g_in (dL/dx_in)

        Also calculate dL/dP, where P are the params of this layer.
        And apply optimizer updates for P.
        """
        pass


class ReLU(Layer):
    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, x_in: npt.NDArray) -> npt.NDArray:
        self.x_in = x_in
        x_out = x_in * (x_in > 0)
        return x_out

    def _backward(self, g_out: npt.NDArray) -> npt.NDArray:
        # This layer needs x_in saved, i.e., activation stashing
        g_in = g_out * (self.x_in.T > 0)
        return g_in


class Linear(Layer):
    def __init__(self, name: str, dim_in: int, dim_out: int):
        super().__init__(name)
        # TODO (anujkalia): Better init
        self.W = np.random.randn(dim_out, dim_in).astype(np.float32)

    def forward(self, x_in: npt.NDArray) -> npt.NDArray:
        self.x_in = x_in
        return self.W @ x_in

    def _backward(self, g_out: npt.NDArray) -> npt.NDArray:
        dL_dW = g_out.T @ self.x_in.T  # (d_out, 1) @ (1, d_in) = (d_out, d_in)
        ret = g_out @ self.W  # (1, d_out) @ (d_out, d_in) = (1, d_in)
        self.W = self.W - (LEARNING_RATE * dL_dW)

        return ret


class LossFn:
    def __init__(self, name: str):
        self.name = name
        self.ground_truth = np.array([])

    def forward(self, x_in: npt.NDArray, ground_truth: npt.NDArray) -> npt.NDArray:
        self.x_in = x_in
        self.ground_truth = ground_truth

        delta = ground_truth - x_in
        return np.array([np.sum(delta**2)])

    def backward(self, g_out: npt.NDArray) -> npt.NDArray:
        ret = 2 * (self.x_in - self.ground_truth)
        ret = ret.reshape(1, N_CLASSES)
        # print(f"Gradient out of {self.name} = {ret}")
        return ret


class MnistClassifier:
    def __init__(self):
        self.matmul1 = Linear(name="matmul1", dim_in=IMG_DIM, dim_out=MLP_HIDDEN_DIM)
        self.relu1 = ReLU(name="relu1")

        self.matmul2 = Linear(name="matmul2", dim_in=MLP_HIDDEN_DIM, dim_out=N_CLASSES)
        self.relu2 = ReLU(name="relu2")

        self.loss_fn = LossFn(name="loss_fn")

    def forward(
        self, image: npt.NDArray, label: int, step: int
    ) -> Tuple[int, np.float32]:
        x = self.matmul1.forward(x_in=image)
        x = self.relu1.forward(x_in=x)
        x = self.matmul2.forward(x_in=x)
        x = self.relu2.forward(x_in=x)

        predicted_label = int(np.argmax(x))

        ground_truth = np.zeros((N_CLASSES, 1), dtype=np.float32)
        ground_truth[label, 0] = 1
        x = self.loss_fn.forward(x_in=x, ground_truth=ground_truth)

        return predicted_label, np.float32(x.item())

    def forward_backward(
        self, image: npt.NDArray, label: int, step: int
    ) -> Tuple[int, np.float32]:
        predicted_label, loss = self.forward(image, label, step)

        g = self.loss_fn.backward(g_out=np.array([]))
        g = self.relu2.backward(g_out=g)
        g = self.matmul2.backward(g_out=g)
        g = self.relu1.backward(g_out=g)
        g = self.matmul1.backward(g_out=g)

        return predicted_label, loss


def test_accuracy(mnist_classifier: MnistClassifier) -> float:
    """
    Return a fraction representing accuracy
    """
    test_images, test_labels = load_test_data()
    num_passed = 0
    for i in range(test_images.shape[0]):
        image_fp32 = test_images[i].astype(np.float32)
        image_fp32 = image_fp32 / image_fp32.sum()

        predicted_label, _ = mnist_classifier.forward(
            image_fp32, label=test_labels[i], step=i
        )
        if predicted_label == test_labels[i]:
            num_passed += 1

    return num_passed / test_images.shape[0]


if __name__ == "__main__":
    np.random.seed(42)

    images, labels = load_training_data()
    c = MnistClassifier()
    test_accuracy(c)

    n_images = images.size
    for i in range(images.shape[0]):
        image_fp32 = images[i].astype(np.float32)
        image_fp32 = image_fp32 / image_fp32.sum()
        c.forward_backward(image_fp32, label=labels[i], step=i)

    print(f"test accuracy = {test_accuracy(c)}")
