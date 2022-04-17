from pprint import pprint
import numpy as np
import dockernn.nn as nn
import dockernn.optim as optim

X = np.array([[0,0, 0],[0,0,1],[0,1,0],[0,1,1], [1,0,0],[1,0,1],[1,1,0],[1,1,1]]).reshape(-1,3)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1])

print(y)

class Model(nn.Module):
    def __init__(self, in_features=3, out_features=1) -> None:
        super().__init__()

        self.ln1 = nn.Linear(in_features=in_features, out_features=50, bias=True)
        self.act1 = nn.ReLU()
        self.ln2 = nn.Linear(in_features=50, out_features=100, bias=True)
        self.act2 = nn.ReLU()
        self.ln3 = nn.Linear(in_features=100, out_features=out_features, bias=True)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.ln3(x)
        x = self.act3(x)

        return x

loss_fn = nn.MSELoss()
model = Model()

model.register_parameters()
pprint(model._parameters, indent=4)
optimizer = optim.SGD(model.parameters(), lr=9.5e-3)
pprint(model._parameters, indent=4)
indices = np.arange(len(y))
iterations = 20000

losses = []
for i in range(iterations):
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    out = model(X)
    optimizer.zero_grad()
    loss = loss_fn(out, y)
    grad = loss_fn.backward()
    model.backward(grad)
    optimizer.step()
    losses.append(loss)
    if i%10==0:
      print(f"it: {i} | loss: {loss}")

test_x = np.array([[0,1,1],[1,1,1],[0,1,0]]).reshape(-1,3)
predictions = model(test_x)
hard_predicitons = np.where(predictions>=0.5,1.0,0.0)

print("Soft Predictions : ", predictions.reshape(-1))
print("Hard Predictions : ", hard_predicitons.reshape(-1))