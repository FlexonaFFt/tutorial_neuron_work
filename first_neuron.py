import numpy as np

# Прописываем сигмоиду для сети
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Прописываем первый нейрон 
class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

weights = np.array([0, 1]) 
bias = 4                
n = Neuron(weights, bias)

x = np.array([2, 3])    
print(n.feedforward(x))
