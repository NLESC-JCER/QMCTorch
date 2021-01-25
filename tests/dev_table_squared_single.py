import torch
from torch import trace, inverse


# total matrix
Atot = 0.1*torch.rand(4, 8)
Btot = 0.1*torch.rand(4, 8)

# occupied state
A = Atot[:, :4]
B = Btot[:, :4]
invA = inverse(A)

# virtual state
Atilde = Atot[:, 4:]
Btilde = Btot[:, 4:]

# excited matrices
Abar = Atot[:, [0, 4, 2, 3]]
Bbar = Btot[:, [0, 4, 2, 3]]
invAbar = inverse(Abar)
out = trace(invAbar@Bbar@invAbar@Bbar)

# M matrix !
Mtilde = invA@Btilde - invA@B@invA@Atilde

# excitatin matrix
mat_exc = invA@Atilde

# index of the
idx_occ = 1
idx_virt = 0
idx = idx_occ*mat_exc.shape[1]+idx_virt

# get the invers of the excitation matrix
T = 1. / mat_exc.view(-1)[idx]

# calue of the the first term
vals = (T * Mtilde.view(-1)[idx])
vals = vals*vals

# compute Y
Y = invA@B@Mtilde
vals += 2 * T*Y.view(-1)[idx]

vals += trace(invA@B@invA@B)

print(out)
print(vals)
