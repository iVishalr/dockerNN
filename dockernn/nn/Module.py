import numpy as np
import inspect
from typing import Callable, Dict, List, Optional, OrderedDict, Union
from .Parameter import Parameter

class Module:
    def __init__(self) -> None:
        self.training = True
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._modules: OrderedDict[str, Optional["Module"]] = OrderedDict()
        self.register: bool = True

    def zero_grad(self, set_to_none: bool = False):
        for _, param in self.active_parameters.items():
            param.zero_grad(set_to_none = set_to_none)

    def type(self, dtype: Union[np.dtype,str]):
        pass

    def parameters(self) -> OrderedDict[str, Parameter]:
        parameters: OrderedDict[str, Parameter] = OrderedDict()
        name_list, parameter_list = self._parameters_()
        for name, param in zip(name_list, parameter_list):
            parameters[name] = param

        self.active_parameters: OrderedDict[str, Parameter] = parameters
        return parameters

    def _parameters_(self) -> List:
        name_list = list()
        param_list = list()
        for name, params in self._parameters.items():
            if isinstance(params, OrderedDict):
                n_list, p_list = self._modules[name]._parameters_()
                n_list = [name+"."+i for i in n_list]
                name_list.extend(n_list)
                param_list.extend(p_list)
            elif isinstance(params, Parameter) and params.requires_grad:
                name_list.append(name)
                param_list.append(params)
        return (name_list, param_list) 

    def named_modules(self) -> OrderedDict[str, "Module"]:
        modules: OrderedDict[str, Module] = OrderedDict()
        for attribute, value in vars(self).items():
            if isinstance(value, Module):
                modules[attribute] = value
        return modules

    def register_parameters(self):
        modules: OrderedDict[str, Module] = self.register_modules()
        if len(list(modules.keys())) == 0:
            return self.get_parameters()

        for key in modules:
            self._modules[key].register_parameters()
            self._parameters[key] = self._modules[key]._parameters

        for name, param in vars(self).items():
            if isinstance(param, Parameter):
                self._parameters[name] = param
        self.register = False
            
    def get_parameters(self) -> OrderedDict[str, Parameter]:
        parameters = OrderedDict()
        for name, param in vars(self).items():
            if isinstance(param, Parameter):
                parameters[name] = param
        self._parameters = parameters
        return parameters

    def register_modules(self) -> OrderedDict[str, "Module"]:
        modules = OrderedDict()
        members = vars(self).items()

        for attribute, value in members:
            if isinstance(value, Module):
                modules[attribute] = value
                value.register_modules()

        self._modules = modules
        return modules

    def train(self, mode: bool = True):
        self._training_mode(mode=mode)
        for name, modules in self._modules.items():
            modules.train(mode=mode)

    def eval(self, mode: bool = False):
        self.train(mode = mode)

    def _training_mode(self, mode: bool = True):
        self.training = mode

    def load_state_dict(self):
        pass

    def state_dict(self):
        pass

    def set_ip(self, ip):
        modules = list(self._modules.keys())
        self.ip = ip
        for name in modules:
            self._modules[name].set_ip(ip)

    def backward(self, grad: np.ndarray = None):
        modules = list(self._modules.keys())[::-1]
        for name in modules[:-1]:
            grad = self._modules[name].backward(grad)
        grad = self._modules[modules[-1]].backward(grad, stop_grad = True)
        return grad

    def _call_implementation(self, *args, **kwargs):
        forward = self.forward
        func_parameters = inspect.signature(forward).parameters
        func_parameters_dict = dict(func_parameters)
        if len(args) > len(func_parameters.keys()) and kwargs is None:
            print(f"Error: Expected to pass {len(func_parameters.keys())} arguments to forward but got {len(args)}.")
            return
        elif len(kwargs.keys()) > len(func_parameters.keys()):
            print(f"Error: Expected to pass {len(func_parameters.keys())} arguments to forward but got {len(kwargs.items())}.")
            return
        func_parameters_keys = list(func_parameters.keys())

        arg_dict = {}
        if args is not None:
            for i in range(0, len(args)):
                func_parameters_dict[func_parameters_keys[i]] = args[i]
                arg_dict[func_parameters_keys[i]] = args[i]

        if kwargs is not None:
            for key,param in func_parameters.items():
                if param.name not in kwargs and param.name not in arg_dict:
                    if not param.default is inspect._empty:
                        func_parameters_dict[key] = param.default
                    else:
                        print(f"Error: Expected to pass {len(func_parameters.keys())} arguments to forward() but got {len(args)+len(kwargs.items())}.")
                        return
                else:
                    if param.name in kwargs:
                        func_parameters_dict[key] = kwargs[key] 
        # print(type(self))
        # if self.register:
        #     self.register_parameters()
        return forward(**func_parameters_dict)

    __call__ = _call_implementation