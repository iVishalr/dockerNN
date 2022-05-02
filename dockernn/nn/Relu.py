import numpy as np
import requests
import json
import blosc

from .Module import Module

class ReLU(Module):
    def __init__(self) -> None:
        self.input_matrix = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_matrix = blosc.pack_array(x)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "act_fn": "relu"
                  }
        r = requests.post(f"http://{self.ip}:30001/forward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = blosc.pack_array(grad)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "grad": grad.hex(), 
                  "act_fn": "relu",
                  }
        r = requests.post(f"http://{self.ip}:30001/backward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

if __name__ == "__main__":
    relu = ReLU()
    a = np.random.randn(5,5)
    out_forward = relu(a)
    out_backward = relu.backward(np.ones(1))
    print(out_forward)
    print(out_backward)

    out = np.ones(1) * np.where(a >= 0, 1.0, 0.0)
    print(out)