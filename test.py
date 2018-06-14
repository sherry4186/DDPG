#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


class NN(chainer.Chain):
    def __init__(self, insize=2, outsize=1):
        super().__init__(
            # bn1=L.BatchNormalization(insize),
            layer1=L.Linear(insize, 5),
            layer2=L.Linear(5, 2),
            layer3=L.Linear(2, outsize)
        )

    def __call__(self, x, norm=False):
        # h = F.leaky_relu(self.bn1(x))
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        h = self.layer3(h)

        if norm:
            h = F.tanh(h)

        return h


n = NN()

x = np.array([[10, 10]], dtype=np.float32)

t = np.array([[100]], dtype=np.float32)
optimizer = chainer.optimizers.Adam()
optimizer.setup(n)


for i in range(1000):
    n.cleargrads()
    y = n(x)
    loss = F.mean_squared_error(y, chainer.Variable(t))
    loss.backward()
    optimizer.update()
    print('i:', i, 'y:', y.data, 'loss:', loss.data, 'grad:', n.layer2.W.data)

print(n(x))


