import numpy as np
import requests
import blosc
from .Module import Module

class Sigmoid(Module):
    def __init__(self) -> None:
        self.input_matrix = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_matrix = blosc.pack_array(x)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "act_fn": "sigmoid"
                  }
        r = requests.post(f"http://{self.ip}:30000/forward", data=params)
        data = r.json()
        self.output = blosc.unpack_array(bytes.fromhex(data["out"]))
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = blosc.pack_array(grad)
        
        params = {"input_matrix": blosc.pack_array(self.output).hex(), 
                  "act_fn": "sigmoid", 
                  "grad": grad.hex(), 
                }
        r = requests.post(f"http://{self.ip}:30000/backward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

if __name__ == "__main__":
    sigmoid = Sigmoid()
    a = np.random.randn(5,5)
    out_forward = sigmoid(a)
    out_backward = sigmoid.backward(np.ones(1))
    print(out_forward)
    print(out_backward)