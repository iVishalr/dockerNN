import os
import numpy as np
import pickle
import dockernn.nn as nn
import dockernn.optim as optim

class CIFAR10:
    def __init__(self,root,train=True):
            self.root = root
            self.split = train
            
            self.data = []
            self.targets = []
            self.train_data = [file for file in os.listdir(root) if "data_batch" in file]
            self.test_data = [file for file in os.listdir(root) if "test_batch" in file]
                    
            data_split = self.train_data if self.split else self.test_data
            
            for files in data_split:
                entry = self.extract(os.path.join(root,files))
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
                    
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))
            self.load_meta()
            
    def extract(self,filename):
        with open(filename,"rb") as f:
            batch_data = pickle.load(f,encoding="latin1")
        return batch_data  
    
    def load_meta(self):
        path = os.path.join(self.root,"batches.meta")
        with open(path,"rb") as infile:
            data = pickle.load(infile,encoding="latin1")
            self.classes = data["label_names"]
            self.classes_to_idx = {_class:i for i,_class in enumerate(self.classes)}

train_dataset = CIFAR10(root="./data/cifar10", train=True)
# test_dataset = CIFAR10(root="./data/cifar10", train=False)

class Model(nn.Module):
    def __init__(self, in_features=3072, out_features=10) -> None:
        super().__init__()
        self.ln1 = nn.Linear(in_features=in_features, out_features=256)
        self.act1 = nn.ReLU()
        self.ln2 = nn.Linear(in_features=256, out_features=128)
        self.act2 = nn.ReLU()
        self.ln3 = nn.Linear(in_features=128, out_features=64)
        self.act3 = nn.ReLU()
        self.ln4 = nn.Linear(in_features=64, out_features=10)
    
    def forward(self, x):
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.ln3(x)
        x = self.act3(x)
        logits = self.ln4(x)
        return logits

IP = "localhost"

loss_fn = nn.CrossEntropyLoss(IP)

model = Model()
model.register_parameters()
model.set_ip(IP)

optimizer = optim.SGD(model.parameters(), lr=3e-3)
BATCH_SIZE = 128
for it in range(10):
    losses = []
    for i,batch in enumerate(range(0, 50000, BATCH_SIZE)):
        X = train_dataset.data[batch: batch+BATCH_SIZE]
        y = train_dataset.targets[batch: batch+BATCH_SIZE]
        X = X/255.0
        X = X.reshape(-1, 32 * 32 * 3)
        y = np.array(y)

        optimizer.zero_grad()
        logits = model(X)

        loss = loss_fn(logits, y)
        grad = loss_fn.backward()
        model.backward(grad)

        optimizer.step()
        print(f"Epoch: {it} it: {i} | loss: {loss.reshape(-1)}")
        losses.append(loss.reshape(-1))
    print(losses)