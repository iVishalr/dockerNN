from typing import Dict
import numpy as np
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/step", methods=["POST"])
def step():

    parameters = request.form["parameters"]
    gradients = request.form["gradients"]
    caches = request.form["cache"]

    parameters: Dict = json.loads(parameters)
    gradients: Dict = json.loads(gradients)
    caches: Dict = json.loads(caches)

    for i in parameters.keys():
        parameters[i] = np.asarray(parameters[i])
        gradients[i] = np.asarray(gradients[i])
        caches[i] = np.asarray(caches[i])

    lr = float(request.form["lr"])
    epsilon = float(request.form["epsilon"])

    for key in caches:
        caches[key] += gradients[key]*gradients[key]

    params, grads, cache = list(parameters.values()), list(gradients.values()), list(caches.values())

    for param, dparam, _cache in zip(params, grads, cache):
        param = param - lr * (dparam)/np.sqrt(_cache + epsilon)
    
    for key,param in zip(list(parameters.keys()), params):
        parameters[key] = param.tolist()

    for key, _cache in zip(list(parameters.keys()), cache):
        caches[key] = _cache.tolist()

    gradients = None
    grads = None

    res = {"parameters": json.dumps(parameters), "cache": json.dumps(caches)}
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30008)