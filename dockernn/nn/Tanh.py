import numpy as np
import blosc
import requests

from .Module import Module

class Tanh(Module):
    def __init__(self) -> None:
        self.input_matrix = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_matrix = blosc.pack_array(x)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "act_fn": "tanh"
                  }
        r = requests.post(f"http://{self.ip}:30002/forward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = blosc.pack_array(grad)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "grad": grad.hex(), 
                  "act_fn": "tanh", 
                  }
        r = requests.post(f"http://{self.ip}:30002/backward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

if __name__ == "__main__":
    tanh = Tanh()
    a = np.random.randn(5,5)
    out_forward = tanh(a)
    out_backward = tanh.backward(np.ones(1))

    print(out_forward)
    print(out_backward)