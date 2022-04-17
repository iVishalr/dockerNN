
import numpy as np
import requests 
import blosc

from .Module import Module

class Softmax(Module):
    def __init__(self) -> None:
        self.output = None

    def forward(self, x: np.ndarray, axis: int) -> np.ndarray:
        input_matrix = blosc.pack_array(x)
        self.axis = axis
        params = {"input_matrix": input_matrix.hex(), 
                  "act_fn": "softmax", 
                  "axis":axis
                  }
        r = requests.post("http://localhost:30003/forward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        self.output = out
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = blosc.pack_array(grad)
        output = blosc.pack_array(self.output)
        params = {"input_matrix": output.hex(), 
                  "act_fn": "softmax", 
                  "axis": self.axis, 
                  "grad": grad.hex(), 
                  }
        r = requests.post("http://localhost:30003/backward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

if __name__ == "__main__":
    softmax = Softmax()
    a = np.random.randn(3,1,10)
    out_forward = softmax(a, axis=-1)
    out_backward = softmax.backward(np.ones(1))
    print(out_forward)
    print(out_backward)