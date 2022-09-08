import torch
import functorch as ft



primal0 = torch.randn(10, 10)
tangent0 = torch.randn(10, 10)
primal1 = torch.randn(10, 10)
tangent1 = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# Here is a basic example to compute the JVP of the above function.
# The jvp(func, primals, tangents) returns func(*primals) as well as the
# computed jvp. Each primal must be associated with a tangent of the same shape.
primal_out, tangent_out = ft.jvp(fn, (primal0, primal1), (tangent0, tangent1))

# functorch.jvp requires every primal to be associated with a tangent.
# If we only want to associate certain inputs to `fn` with tangents,
# then we'll need to create a new function that captures inputs without tangents:
primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)
y = torch.randn(10, 10)

import functools
new_fn = functools.partial(fn, y=y)
primal_out, tangent_out = ft.jvp(new_fn, (primal,), (tangent,))

from functorch import jvp
x = torch.randn(5)
y = torch.randn(5)

import torch.nn as nn
f = lambda x, y: (x * y)
_, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
assert torch.allclose(output, x + y)
lin = nn.Linear(1, 1)
input = torch.randn(2, 1)
print('output', lin(input))
tangent = torch.ones(2, 1)


from torch.optim import Adam
optimizer = Adam(lin.parameters())
with torch.enable_grad():
    optimizer.zero_grad()
    output = lin(input)
    l = output.sum()
    l.backward()
    optimizer.step()

torch.ones(1)


print('ok')