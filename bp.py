"""
How does backprop through langevin sampling work?
"""

import torch
import torch.nn as nn

# a single scalar wrapped in a torch module
logp_net = nn.Sequential(nn.Linear(1, 1, bias=False))


def print_params(params):
    return ' '.join([str(p.item()) for p in params])


print(f"logp_net params: {print_params(logp_net.parameters())}")

x = torch.tensor([2.5])
x.requires_grad_(True)

print(f"x: {x.item()}")

f_prime = torch.autograd.grad(logp_net(x).sum(), [x], retain_graph=True, create_graph=True)[0]

print(f"grad: {f_prime.item()}")

noise = torch.randn_like(x)
x_kl = x + 1.0 * f_prime + .01 * noise

loss_grad = logp_net(x_kl)

logp_net.requires_grad_(False)
loss_nograd = logp_net(x_kl)
logp_net.requires_grad_(True)

print(f"loss grad: {loss_grad.item()}")
print(f"loss nograd: {loss_nograd.item()}")
assert torch.isclose(loss_grad, loss_nograd)

loss_grad.backward(retain_graph=True)

print(f"loss grad wrt params: {print_params([p.grad for p in logp_net.parameters()])}")

loss_nograd.backward(retain_graph=True)

print(f"loss nograd wrt params: {print_params([p.grad for p in logp_net.parameters()])}")

print(f"x_kl: {x_kl.item()}")
