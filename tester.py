#%%
import tensorflow as tf

x = tf.cast([[[[[1]]]]], tf.int32)
y = x
x += 1

print(x, y)

# %%
import tensorflow as tf
from keras import layers, Model

class testtt(Model):
    def __init__(self):
        super().__init__()
        # self.layer = layers.BatchNormalization()
        # self.layer2 = layers.LayerNormalization(epsilon=0.00001)
        # self.relu = layers.ReLU()
        self.dropout = layers.Dropout(rate=0.5)
    
    def call(self, x):
        # x = self.layer(x, training=True)
        # x = self.layer2(x)
        # x = self.relu(x)
        x = self.dropout(x)
        return x
    
j = testtt()
i = tf.random.uniform([5, 32, 32, 3])
i = tf.random.uniform([3, 2])
ans = tf.zeros([3, 2])
opt = tf.optimizers.SGD()
print(i)
print(j(i))
# printweight(j.weights)
# j.compile(optimizer="sgd", loss="mse")
# j.fit(i, ans, epochs=2)

# with tf.GradientTape() as tape:
#     k = j(i)
#     loss = tf.losses.mean_squared_error(k, ans)
# grad = tape.gradient(loss, j.trainable_weights)
# opt.apply_gradients(zip(grad, j.trainable_weights))
# print(k)

def printweight(weight):
    print("##")
    for w in weight:
        print(w)

printweight(j.weights)
printweight(j.trainable_weights)

# %%
import torch
layer = torch.nn.BatchNorm1d(2)
x = torch.rand([2, 3, 3])
y = layer(x)
print(x)
print(y)

# %%
import numpy as np
a = np.random.random([4, 5, 6])
for i in a:
    print(i)
# %%

def jj(a, b, c):
    print(a+b+c)

def jojo(d, **kwargs):
    print(d)
    jj(**kwargs)

dic = {'a':1, 'b':2, 'c':4, 'd':8}

jojo(**dic)
# %%
