from engine import Value
from nn import Module,MLP,Layer,Neuron

model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))