import json
from typing import Dict
import numpy as np
import time
import requests

from ..nn import Parameter

class Adagrad:
    def __init__(self, parameters: Dict[str, Parameter], lr: float = 3e-4, epsilon: float = 1e-8) -> None:
        self.parameters = parameters
        self.lr = lr
        self.epsilon = epsilon
        self.cache = {key:np.zeros_like(param.tensor) for key,param in self.parameters.items()}

    def zero_grad(self, set_to_none: bool = False) -> None:
        for i in self.parameters:
            self.parameters[i].zero_grad(set_to_none=set_to_none)

    def step(self):
        parameters = {}
        gradients = {}
        cache = {}

        for name, param in self.parameters.items():
            parameters[name] = param.tensor.tolist()
            gradients[name] = param.gradient.tolist()
            cache[name] = self.cache[name].tolist()

        params = {"parameters": json.dumps(parameters), "gradients": json.dumps(gradients), "cache": json.dumps(cache) ,"lr": self.lr, "epsilon": self.epsilon}
        r = requests.post("http://localhost:10001/step", data=params)
        data = r.json()
        
        parameters = json.loads(data["parameters"])
        cache = json.loads(data["cache"])

        for name in self.parameters.keys():
            self.parameters[name].tensor = np.asarray(parameters[name])
            self.cache[name] = np.asarray(cache[name])

if __name__ == "__main__":
    w = Parameter(np.random.randn(128,256), requires_grad=True)
    w.gradient = np.random.randn(128,256)

    parameters = {"Linear1": w, "Linear2": w, "Linear3": w, "Linear4": w}
    adagrad = Adagrad(parameters, 0.001)
    start = time.time()
    for i in range(0,1):
        adagrad.step()
    print(f"Time take to optimze for 1 iterations : {time.time() - start}s.")