import numpy as np
import json
import blosc
from flask import Flask, jsonify, request

app = Flask(__name__)        

@app.route("/forward", methods=["POST"])
def forward():
    input_matrix = blosc.unpack_array(bytes.fromhex(request.form['input_matrix']))
    sigmoid = request.form['act_fn']
    
    if sigmoid == "sigmoid":
        out = 1.0/(1+np.exp(-input_matrix))
        out = blosc.pack_array(out)
        res = {"out": out.hex(),"act_fn": sigmoid}
        return jsonify(res)
    else:
        print(f"Error: Expected act_fn to be sigmoid but found {sigmoid}.")


@app.route("/backward", methods=["POST"])
def backward():
    input_matrix = blosc.unpack_array(bytes.fromhex(request.form["input_matrix"]))
    grad = blosc.unpack_array(bytes.fromhex(request.form["grad"]))

    act_fn = request.form['act_fn']

    if act_fn=="sigmoid":
        out = grad * input_matrix * (1-input_matrix)
        out = blosc.pack_array(out)
        res = {"out": out.hex(), "act_fn": act_fn}
        return jsonify(res)
    else:
        print(f"Error: Expected act_fn to be sigmoid but found {act_fn}.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30000)