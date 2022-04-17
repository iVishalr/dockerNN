import numpy as np
from flask import Flask, request, jsonify
import blosc
import json

app = Flask(__name__)

@app.route("/forward", methods=["POST"])
def forward():
    weights = request.form["weights"]
    bias = request.form["bias"]   
    input_matrix = request.form["input_matrix"]

    # weights are of shape (out_features, in_features)
    # inputs are of shape (B, -1, in_features)
    # bias is of shape (1,)
    # print(weights)

    weights = blosc.unpack_array(bytes.fromhex(weights))
    if bias != "None":
        bias = blosc.unpack_array(bytes.fromhex(bias))
    else:
        bias = 0

    input_matrix = blosc.unpack_array(bytes.fromhex(input_matrix))

    z = input_matrix @ weights.T + bias
    z = blosc.pack_array(z).hex()
    res = {"out": z}
    return jsonify(res)

@app.route("/backward", methods=["POST"])
def backward():
    input_matrix = request.form["input_matrix"]
    weights = request.form["weights"]
    grad = request.form["grad"]
    use_bias = request.form["use_bias"]
    stop_grad = request.form["stop_grad"]

    dW = request.form["dW"]
    db = request.form["db"]
    
    input_matrix = blosc.unpack_array(bytes.fromhex(input_matrix))
    weights = blosc.unpack_array(bytes.fromhex(weights))
    grad = blosc.unpack_array(bytes.fromhex(grad))

    dW = blosc.unpack_array(bytes.fromhex(dW))
    db = blosc.unpack_array(bytes.fromhex(db))
    
    if len(grad.shape) == 2:
        wgrad = grad.T 
    else:
        wgrad = grad.transpose(0,-1,-2)
    
    # accumulate grads in parameters
    # input : (8, 1, 3)
    # w : 50,3
    # grad = (8, 50)
    # W_grad = (50, 8) @ (8, 3)
    dW += np.sum(grad.T @ input_matrix, axis=0, keepdims=True)
    db += np.sum(grad) if use_bias == "True" else np.zeros(1)
    
    if stop_grad == "False":
        # grad = (8,100)
        # weights = (100,50)
        # x = (8,50)
        print("false")
        dx = grad @ weights
    else:
        # print("Assigning zero grad")
        dx = np.zeros_like(input_matrix)

    dW = blosc.pack_array(dW)
    db = blosc.pack_array(db)
    dx = blosc.pack_array(dx)
    
    res = {"dW": dW.hex(), "db": db.hex(), "dx": dx.hex()}
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=20000)