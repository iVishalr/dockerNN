import json
from typing import Dict
import numpy as np
import time
import requests

from ..nn import Parameter

class SGD:
    def __init__(self, parameters: Dict[str, Parameter], lr: float = 3e-4, ip:str = "localhost") -> None:
        self.parameters = parameters
        self.lr = lr
        self.ip = ip
    
    def zero_grad(self, set_to_none: bool = False):
        for i in self.parameters:
            self.parameters[i].zero_grad(set_to_none=set_to_none)

    def step(self):
        parameters = {}
        gradients = {}
        # print(self.parameters)
        for name, param in self.parameters.items():
            parameters[name] = param.tensor.tolist()
            gradients[name] = param.gradient.tolist()

        params = {"parameters": json.dumps(parameters), "gradients": json.dumps(gradients),"lr": self.lr}
        r = requests.post(f"http://{self.ip}:30007/step", data=params)
        data = r.json()
        parameters = json.loads(data["parameters"])
        
        for name in self.parameters.keys():
            self.parameters[name].tensor = np.asarray(parameters[name])

if __name__ == "__main__":
    w = Parameter(np.random.randn(128,256), requires_grad=True)
    w.gradient = np.random.randn(128,256)

    parameters = {"Linear1": w, "Linear2": w, "Linear3": w, "Linear4": w}
    sgd = SGD(parameters, 0.001)
    start = time.time()
    for i in range(0,1):
        sgd.step()
    print(f"Time take to optimze for 1 iterations : {time.time() - start}s.")