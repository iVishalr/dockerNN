from typing import Dict
import numpy as np
import json
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/step", methods=["POST"])
def step():

    parameters = request.form["parameters"]
    gradients = request.form["gradients"]

    parameters: Dict = json.loads(parameters)
    gradients: Dict = json.loads(gradients)

    for i in parameters.keys():
        parameters[i] = np.asarray(parameters[i])
        gradients[i] = np.asarray(gradients[i])
    
    lr = float(request.form["lr"])

    params, grads = list(parameters.values()), list(gradients.values())

    for key, param, dparam in zip(list(parameters.keys()),params, grads):
        # print(f"********\n{param}\n----------")
        param = param - lr * dparam
        parameters[key] = param.tolist()
        # print(f"{param}\n******")
    
    # for key,param in zip(list(parameters.keys()), params):
    #     parameters[key] = param.tolist()

    gradients = None
    grads = None

    res = {"parameters": json.dumps(parameters)}
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30007)