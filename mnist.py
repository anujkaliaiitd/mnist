from data_loader import load_training_data, load_test_data
import os
from abc import ABC, abstractmethod
from typing import Tuple
import torch


IMG_DIM = 784
MLP_HIDDEN_DIM = 128
N_CLASSES = 10
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.035"))


class Layer(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Given x_in, return x_out
        """
        pass

    def backward(self, g_out: torch.Tensor) -> torch.Tensor:
        """
        Shared logic that must always run first.
        """
        # print(f"g_out for {self.name} = {g_out}")
        return self._backward(g_out)

    @abstractmethod
    def _backward(self, g_out: torch.Tensor) -> torch.Tensor:
        """
        Given g_out (dL/dx_out) return g_in (dL/dx_in)

        Also calculate dL/dP, where P are the params of this layer.
        And apply optimizer updates for P.
        """
        pass


class ReLU(Layer):
    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        self.x_in = x_in
        x_out = x_in * (x_in > 0)
        return x_out

    def _backward(self, g_out: torch.Tensor) -> torch.Tensor:
        # This layer needs x_in saved, i.e., activation stashing
        g_in = g_out * (self.x_in.T > 0)
        return g_in


class Linear(Layer):
    def __init__(self, name: str, dim_in: int, dim_out: int):
        super().__init__(name)
        # TODO (anujkalia): Better init
        self.W = torch.randn(dim_out, dim_in, dtype=torch.float32)
        self.B = torch.randn(dim_out, 1, dtype=torch.float32)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        self.x_in = x_in
        ret = (self.W @ x_in) + self.B
        return ret

    def _backward(self, g_out: torch.Tensor) -> torch.Tensor:
        dL_dW = g_out.T @ self.x_in.T  # (d_out, 1) @ (1, d_in) = (d_out, d_in)
        dL_dB = g_out

        ret = g_out @ self.W  # (1, d_out) @ (d_out, d_in) = (1, d_in)
        self.W = self.W - (LEARNING_RATE * dL_dW)
        self.B = self.B - (LEARNING_RATE * dL_dB.T)

        return ret

class LossFn:
    """
    Softmax + cross-entropy
    """
    def __init__(self, name: str):
        self.name = name
        self.ground_truth = torch.empty(0, dtype=torch.float32)

    def forward(self, x_in: torch.Tensor, label: int) -> torch.Tensor:
        self.label = int(label)

        x = x_in - torch.max(x_in)
        exp_x = torch.exp(x)
        self.probs = exp_x / torch.sum(exp_x)

        ret = -torch.log(self.probs[self.label])
        return ret

    def backward(self, g_out: torch.Tensor) -> torch.Tensor:
        ret = self.probs.clone()
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
        self, image: torch.Tensor, label: int, step: int
    ) -> Tuple[int, float]:
        x = image
        x = self.matmul1.forward(x_in=x)
        x = self.relu1.forward(x_in=x)
        x = self.matmul2.forward(x_in=x)
        # x = self.relu2.forward(x_in=x)

        predicted_label = int(torch.argmax(x))

        x = self.loss_fn.forward(x_in=x, label=label)

        return predicted_label, float(x.item())

    def forward_backward(
        self, image: torch.Tensor, label: int, step: int
    ) -> Tuple[int, float]:
        predicted_label, loss = self.forward(image, label, step)

        g = self.loss_fn.backward(g_out=torch.empty(0, dtype=torch.float32))
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
        image_fp32 = torch.tensor(test_images[i], dtype=torch.float32)
        image_fp32 = image_fp32 / 255.0

        predicted_label, _ = mnist_classifier.forward(
            image_fp32, label=test_labels[i], step=i
        )
        if predicted_label == test_labels[i]:
            num_passed += 1

    return num_passed / test_images.shape[0]


if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"learning rate = {LEARNING_RATE}")

    images, labels = load_training_data()
    c = MnistClassifier()

    n_images = images.size

    for passes in range(2):
        for i in range(images.shape[0]):
            image_fp32 = torch.tensor(images[i], dtype=torch.float32)
            image_fp32 = image_fp32 / 255.0
            c.forward_backward(image_fp32, label=labels[i], step=i)
        print(f"test accuracy after {passes} passes = {test_accuracy(c)}")
        permutation = torch.randperm(images.shape[0]).numpy()
        images = images[permutation]
        labels = labels[permutation]
