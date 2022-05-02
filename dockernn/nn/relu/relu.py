import numpy as np
import blosc
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/forward", methods=["POST"])
def forward():
    input_matrix = blosc.unpack_array(bytes.fromhex(request.form['input_matrix']))

    act_fn = request.form['act_fn']

    if act_fn == 'relu':
        out = np.where(input_matrix >= 0, input_matrix, 0.0)
        out = blosc.pack_array(out)
        res = {"out": out.hex(),"act_fn": "relu"}
        return jsonify(res)
    else:
        print(f"Error: Expected act_fn to be relu but got {act_fn}")

@app.route("/backward", methods=["POST"])
def backward():
    input_matrix = blosc.unpack_array(bytes.fromhex(request.form["input_matrix"]))
    grad = blosc.unpack_array(bytes.fromhex(request.form["grad"]))

    act_fn = request.form["act_fn"]

    if act_fn == "relu":
        out = grad * np.where(input_matrix >= 0, 1.0, 0.0)
        out = blosc.pack_array(out)
        res = {"out": out.hex(), "act_fn": "relu"}
        return jsonify(res)
    else:
        print(f"Error: Expected act_fn to be relu but got {act_fn}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30001)