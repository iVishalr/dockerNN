import numpy as np
import json
import blosc
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/forward", methods=["POST"])
def forward():
    input_matrix = blosc.unpack_array(bytes.fromhex(request.form['input_matrix']))

    act_fn = request.form["act_fn"]
    axis = int(request.form["axis"])

    if act_fn == "softmax":
        e = np.exp(input_matrix-input_matrix.max(axis=axis,keepdims=True))
        out = e/np.nansum(e,axis=axis,keepdims=True)
        out = blosc.pack_array(out)
        res = {"out": out.hex(),"act_fn": act_fn}
        return jsonify(res)
    else:
        print(f"Error: Expected act_fn to be softmax but got {act_fn}")

@app.route("/backward", methods=["POST"])
def backward():
    input_matrix = blosc.unpack_array(bytes.fromhex(request.form['input_matrix']))
    grad = blosc.unpack_array(bytes.fromhex(request.form["grad"]))

    act_fn = request.form["act_fn"]
    axis = int(request.form["axis"])
    
    if act_fn == "softmax":
        out = input_matrix * (grad - np.sum(input_matrix * grad, axis=axis, keepdims=True))
        out = blosc.pack_array(out)
        res = {"out": out.hex(),"act_fn": act_fn}
        return jsonify(res)
    else:
        print(f"Error: Expected act_fn to be softmax but got {act_fn}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30003)