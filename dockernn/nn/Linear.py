import numpy as np
import requests 
import blosc

from .Module import Module
from .Parameter import Parameter

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.weights = Parameter(np.random.randn(out_features, in_features) * 0.01)
        self.bias = Parameter(np.zeros(1)) if bias else None

        self.use_bias = bias
        self.input_matrix = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.w = blosc.pack_array(self.weights.tensor)
        self.b = blosc.pack_array(self.bias.tensor) if self.use_bias else None
        self.input_matrix = blosc.pack_array(x)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "weights": self.w.hex(), "bias": self.b.hex(), 
                  }
        r = requests.post(f"http://{self.ip}:30004/forward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

    def backward(self, grad: np.ndarray, stop_grad = False) -> np.ndarray:
        self.dW = self.weights.gradient
        self.db = self.bias.gradient
        self.dW = blosc.pack_array(self.dW)
        self.db = blosc.pack_array(self.db)

        grad_in = blosc.pack_array(grad)
        params = {"input_matrix": self.input_matrix.hex(), 
                  "weights": self.w.hex(), 
                  "grad": grad_in.hex(), 
                  "dW": self.dW.hex(), 
                  "db": self.db.hex(), 
                  "stop_grad": stop_grad, 
                  "use_bias": self.use_bias,
                  }
        r = requests.post(f"http://{self.ip}:30004/backward", data=params)
        data = r.json()
        dW = data["dW"]
        db = data["db"]
        grad = data["dx"]

        self.weights.gradient = blosc.unpack_array(bytes.fromhex(dW))
        self.bias.gradient = blosc.unpack_array(bytes.fromhex(db))
        grad = blosc.unpack_array(bytes.fromhex(grad))
        return grad

if __name__ == "__main__":
    linear = Linear(in_features= 50, out_features = 100)
    x = np.random.randn(64, 1, 50)
    out_forward = linear(x)
    out_backward = linear.backward(np.random.randn(64,1,100))
    print(out_forward)
    print(out_backward)