import numpy as np
import blosc
import requests

class MSELoss:
    def __init__(self, ip: str = "localhost") -> None:
        self.logits = None
        self.targets = None
        self.ip = ip

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.logits = blosc.pack_array(logits)
        self.targets = blosc.pack_array(targets)
        # print(targets)
        params = {"logits": self.logits.hex(), 
                  "targets": self.targets.hex()}
        r = requests.post(f"http://localhost:30005/forward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["loss"]))
        out = out.reshape(-1)
        return out

    def backward(self, grad: np.ndarray = None) -> np.ndarray:
        if grad is not None:
            grad = grad
        else:
            grad = np.ones(1)
        grad = blosc.pack_array(grad)
        params = {"logits": self.logits.hex(), 
                  "targets": self.targets.hex(), 
                  "grad": grad.hex(),
                  }
        r = requests.post(f"http://localhost:30005/backward", data=params)
        data = r.json()
        out = blosc.unpack_array(bytes.fromhex(data["out"]))
        return out

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return self.forward(logits=logits, targets=targets)

if __name__ == "__main__":
    loss_fn = MSELoss()
    a = np.random.randn(64, 1)
    b = np.random.randn(64,)
    loss = loss_fn(a,b)
    print(loss)
    grad = loss_fn.backward()
    print(grad)