import numpy as np
import blosc
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/forward", methods=["POST"])
def forward():
    logits = blosc.unpack_array(bytes.fromhex(request.form["logits"]))
    targets = blosc.unpack_array(bytes.fromhex(request.form["targets"]))

    targets = targets.reshape(-1)
    m = targets.shape[0]

    loss = np.sum(-np.log(logits[range(m), targets])) / m
    loss = loss.reshape(-1,1)
    loss = blosc.pack_array(loss)
    res = {"loss": loss.hex()}
    return jsonify(res)

@app.route("/backward", methods=["POST"])
def backward():
    logits = blosc.unpack_array(bytes.fromhex(request.form["logits"]))
    targets = blosc.unpack_array(bytes.fromhex(request.form["targets"]))

    grad = request.form["grad"]
    grad = blosc.unpack_array(bytes.fromhex(grad))

    targets = targets.reshape(-1)
    m = targets.shape[0]

    logits[range(m), targets] -= 1
    out = grad * logits / m
    out = blosc.pack_array(out)
    res = {"out": out.hex()}
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30006)