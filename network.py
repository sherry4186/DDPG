#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


class NN(chainer.Chain):
    def __init__(self, insize, outsize):
        super().__init__(
            # bn1=L.BatchNormalization(insize),
            layer1=L.Linear(insize, 110),
            layer2=L.Linear(110, 100),
            layer3=L.Linear(100, outsize, initialW=np.zeros((outsize, 100), dtype=np.float32))
        )

    def __call__(self, x, norm=False):
        # h = F.leaky_relu(self.bn1(x))
        h = F.leaky_relu(self.layer1(x))
        h = F.leaky_relu(self.layer2(h))
        h = self.layer3(h)

        if norm:
            h = F.tanh(h)

        return h

    def weight_update(self, w1, new):
        self.layer1.W.data = w1 * new.layer1.W.data + (1 - w1) * self.layer1.W.data
        self.layer2.W.data = w1 * new.layer2.W.data + (1 - w1) * self.layer2.W.data
        self.layer3.W.data = w1 * new.layer3.W.data + (1 - w1) * self.layer3.W.data
        self.layer1.b.data = w1 * new.layer1.b.data + (1 - w1) * self.layer1.b.data
        self.layer2.b.data = w1 * new.layer2.b.data + (1 - w1) * self.layer2.b.data
        self.layer3.b.data = w1 * new.layer3.b.data + (1 - w1) * self.layer3.b.data

    def return_grad(self):
        layer1_w_grad = self.layer1.W.grad
        layer1_b_grad = self.layer1.b.grad
        layer2_w_grad = self.layer2.W.grad
        layer2_b_grad = self.layer2.b.grad
        layer3_w_grad = self.layer3.W.grad
        layer3_b_grad = self.layer3.b.grad
        return layer1_w_grad, layer1_b_grad, layer2_w_grad, layer2_b_grad, layer3_w_grad, layer3_b_grad
