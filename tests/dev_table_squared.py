import torch


Atot = torch.rand(3, 5)
Btot = torch.rand(3, 5)

A0 = Atot[:, :3]
B0 = Btot[:, :3]

Abar = Atot[:, [0, 1, 3]]
Bbar = Btot[:, [0, 1, 3]]

P = torch.zeros(3, 3)
P[2, 2] = 1
