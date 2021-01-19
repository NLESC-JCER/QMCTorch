import torch


Atot = torch.rand(4, 6)
Btot = torch.rand(4, 6)

A = Atot[:, :4]
B = Btot[:, :4]

Abar = Atot[:, [0, 4, 2, 5]]
Bbar = Btot[:, [0, 4, 2, 5]]

P = torch.zeros(4, 4)
P[1, 1] = 1
P[3, 3] = 1

Ax = Abar - A
invA = torch.inverse(A)
