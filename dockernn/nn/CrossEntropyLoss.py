import numpy as np
import json
import blosc
import requests
from .Softmax import Softmax

class CrossEntropyLoss:
    def __init__(self) -> None:
        self.logits = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.logits = blosc.pack_array(logits)
        self.targets = blosc.pack_array(targets)
        params = {"input_matrix": self.logits.hex(), 
                  "act_fn": "softmax", 
                  "axis": -1
                  }
        
        

        r = requests.post("http://localhost:30003/forward", data=params)
        data = r.json()
        self.log_logits = data["out"]

        print(blosc.unpack_array(bytes.fromhex(self.log_logits)))
        print(targets)

        params = {"logits": self.log_logits, "targets": self.targets.hex()}
        r = requests.post("http://localhost:40001/forward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["loss"]))
        return out

    def backward(self, grad: np.ndarray = None) -> np.ndarray:
        if grad is not None:
            grad_in = blosc.compress(grad)
        else:
            grad_in = np.ones(1)
        grad_in = blosc.pack_array(grad_in)
        params = {"logits": self.log_logits, 
                  "targets": self.targets.hex(),
                  "grad": grad_in.hex(),
                 }
        r = requests.post("http://localhost:40001/backward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return self.forward(logits=logits, targets=targets)

if __name__ == "__main__":
    loss_fn = CrossEntropyLoss()
    softmax = Softmax()
    logits = np.random.randn(10,10)
    print(logits)
    targets = np.array([1,0,4,5,9,3,2,7,6,8]).reshape(-1,1)
    softlogits = softmax(logits)
    print(softlogits)
    print(logits.sum(axis=1))
    loss = loss_fn(softlogits, targets)
    print(loss)
    grad = loss_fn.backward()
    print(grad.shape)