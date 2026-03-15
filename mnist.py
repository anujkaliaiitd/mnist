from data_loader import load_training_data, load_test_data
import os
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Tuple


IMG_DIM = 784
MLP_HIDDEN_DIM = 32
N_CLASSES = 10
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1.0"))


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
    """
    Softmax + cross-entropy
    """
    def __init__(self, name: str):
        self.name = name
        self.ground_truth = np.array([])

    def forward(self, x_in: npt.NDArray, label: int) -> np.float32:
        self.label = label

        x = x_in - np.max(x_in)
        exp_x = np.exp(x)
        self.probs = exp_x / np.sum(exp_x)

        ret = -np.log(self.probs[label])
        return ret

    def backward(self, g_out: npt.NDArray) -> npt.NDArray:
        ret = self.probs
        ret[self.label] -= 1
        ret = ret.reshape(1, N_CLASSES)
        return ret


class MnistClassifier:
    def __init__(self):
        self.matmul1 = Linear(name="matmul1", dim_in=IMG_DIM, dim_out=MLP_HIDDEN_DIM)
        self.relu1 = ReLU(name="relu1")

        self.matmul2 = Linear(name="matmul2", dim_in=MLP_HIDDEN_DIM, dim_out=N_CLASSES)
        #self.relu2 = ReLU(name="relu2")

        self.loss_fn = LossFn(name="loss_fn")

    def forward(
        self, image: npt.NDArray, label: int, step: int
    ) -> Tuple[int, np.float32]:
        x = self.matmul1.forward(x_in=image)
        x = self.relu1.forward(x_in=x)
        x = self.matmul2.forward(x_in=x)
        # x = self.relu2.forward(x_in=x)

        predicted_label = int(np.argmax(x))

        x = self.loss_fn.forward(x_in=x, label=label)

        return predicted_label, np.float32(x.item())

    def forward_backward(
        self, image: npt.NDArray, label: int, step: int
    ) -> Tuple[int, np.float32]:
        predicted_label, loss = self.forward(image, label, step)

        g = self.loss_fn.backward(g_out=np.array([]))
        # g = self.relu2.backward(g_out=g)
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
    print(f"learning rate = {LEARNING_RATE}")

    images, labels = load_training_data()
    c = MnistClassifier()
    test_accuracy(c)

    n_images = images.size
    for i in range(images.shape[0]):
        image_fp32 = images[i].astype(np.float32)
        image_fp32 = image_fp32 / image_fp32.sum()
        c.forward_backward(image_fp32, label=labels[i], step=i)

    print(f"test accuracy = {test_accuracy(c)}")
