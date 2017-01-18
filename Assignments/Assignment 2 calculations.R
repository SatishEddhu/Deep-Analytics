# Problem 2: Linear perceptrons
eta = 0.01
x = 1
y = 6
w13 = w23 = wx1 = wx2 = 1

o1 = wx1 * x
o2 = wx2 * x
o3 = (w13 * o1) + (w23 * o2)

Ebyw13 = (o3-y) * o1
Ebyw23 = (o3-y) * o2
Ebywx1 = (o3-y) * w13 * x
Ebywx2 = (o3-y) * w23 * x

w13 = w13 - eta*Ebyw13
w23 = w23 - eta*Ebyw23
wx1 = wx1 - eta*Ebywx1
wx2 = wx2 - eta*Ebywx2

# Problem 3: Sigmoid perceptrons
sigmoid = function(p) {
  1/(1+exp(-p))
}

sigmoidD = function(p) {
  sigmoid(p) * (1 - sigmoid(p))
}

eta = 0.01
x = 0.2
y = 1
w13 = w23 = wx1 = wx2 = 1

a1 = wx1*x
o1 = sigmoid(a1)

a2 = wx2*x
o2 = sigmoid(a2)

a3 = w13*o1 + w23*o2
o3 = sigmoid(a3)

Ebyw13 = (o3-y)*o1*sigmoidD(a3)
Ebyw23 = (o3-y)*o2*sigmoidD(a3)

Ebywx1 = (o3-y)*w13*sigmoidD(a3)*x*sigmoidD(a1)
Ebywx2 = (o3-y)*w23*sigmoidD(a3)*x*sigmoidD(a2)

w13 = w13 - eta*Ebyw13
w23 = w23 - eta*Ebyw23
wx1 = wx1 - eta*Ebywx1
wx2 = wx2 - eta*Ebywx2

