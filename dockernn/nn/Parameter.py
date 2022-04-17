from typing import Tuple, Union, List
import numpy as np

class Parameter:
    def __init__(self, tensor: np.ndarray, requires_grad: bool = True) -> None:
        self.tensor = tensor
        self.gradient = np.zeros_like(tensor)
        self.requires_grad = True
    
    def item(self) -> np.ndarray:
        return self.tensor

    def shape(self) -> Union[List,Tuple]:
        return self.tensor.shape

    def requires_grad(self, bool: bool) -> None:
        self.requires_grad = bool

    def zero_grad(self, set_to_none: bool = False) -> None:
        if set_to_none:
            self.gradient = None
        else:
            self.gradient = np.zeros_like(self.tensor)