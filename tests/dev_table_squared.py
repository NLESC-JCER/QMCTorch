import torch
from torch import trace, inverse


def get_trace(A, B):
    return torch.trace(torch.inverse(A) @ B)


def get_trace_square(A, B):
    mat = inverse(A)@B
    return torch.trace(mat@mat)


def proj(A, Pleft, Pright):
    return Pleft.T@A@Pright


# total matrix
Atot = 0.1*torch.rand(4, 8)
Btot = 0.1*torch.rand(4, 8)

# occupied state
A = Atot[:, :4]
B = Btot[:, :4]

# virtual state
Atilde = Atot[:, 4:]
Btilde = Btot[:, 4:]

# excited matrices
Abar = Atot[:, [0, 4, 2, 6]]
Bbar = Btot[:, [0, 4, 2, 6]]

# size conserving projector
P = torch.zeros(4, 4)
P[1, 1] = 1
P[3, 3] = 1

# projector
PP = torch.zeros(4, 2)
PP[1, 0] = 1
PP[3, 1] = 1

# projector for total matrices
PPtot = torch.zeros(6, 2)
PPtot[1, 0] = 1
PPtot[3, 1] = 1

# shortucts
Ax = Abar - A
Bx = Bbar-B
invA = torch.inverse(A)
I = torch.eye(4)

T = P@inverse(invA@Abar)@P
Z = invA@Ax@T@invA

# reference answer
out = get_trace_square(Abar, Bbar)

# first term
alpha = trace(invA@Bbar@invA@Bbar)

alpha1 = trace(invA@B@invA@B)
alpha2 = trace(invA@(Bbar-B)@P @ invA@(Bbar-B)@P)
alpha3 = trace(invA@B@invA@(Bbar-B)@P)

assert(torch.allclose(alpha, alpha1+alpha2+2*alpha3))


# second term
beta = trace(invA@Bbar@Z@Bbar)

b1_mat = (invA@Bbar)@(invA@Bbar)@invA@Abar @ T
beta1 = trace(b1_mat)

b11 = trace(invA@B@invA@B@invA@Abar@T)
b12 = trace(invA@Bx@invA@Bx)
b13 = trace(invA@B@invA@Bx)
b14 = trace(invA@Bx@invA@B@invA@Abar@T)

assert(torch.allclose(beta1, b11+b12+b13+b14))

b2_mat = (invA@Bbar)@(invA@Bbar) @ T
beta2 = trace(b2_mat)

b21 = trace(invA@B@invA@B@T)
b22 = trace(invA@Bx@invA@Bx@T)
b23 = trace(invA@Bx@invA@B@T)
b24 = trace(invA@B@invA@Bx@T)

assert(torch.allclose(beta2, b21+b22+b23+b24))

assert(torch.allclose(beta, beta1-beta2))

# third term
gamma = trace(Z@Bbar@Z@Bbar)

g1_mat = invA@Abar@T@invA@Bbar
gamma1 = trace(g1_mat@g1_mat)

g11m = invA@Abar@T@invA@B
g11 = trace(g11m@g11m)

g12m = invA@Bx
g12 = trace(g12m@g12m)


g13m = invA@B@invA@Abar@T@invA@Bx
g13 = trace(g13m)

assert(torch.allclose(gamma1, g11+g12+2*g13))

g2_mat = T@invA@Bbar
gamma2 = trace(g2_mat@g2_mat)

g21m = T@invA@B
g21 = trace(g21m@g21m)

g22m = T@invA@Bx
g22 = trace(g22m@g22m)

g23m = T@invA@B@T@invA@Bx
g23 = trace(g23m)

assert(torch.allclose(gamma2, g21+g22+2*g23))

g3_mat = (T@invA@Bbar)@(T@invA@Bbar)@invA@Abar
gamma3 = trace(g3_mat)

g31m = T@invA@B@T@invA@B@invA@Abar
g31 = trace(g31m)

g32m = invA@Bx@invA@Bx@T
g32 = trace(g32m)

g33m = invA@B@T@invA@Bx
g33 = trace(g33m)

g34m = T@invA@Bx@T@invA@B@invA@Abar
g34 = trace(g34m)

assert(torch.allclose(gamma3, g31+g32+g33+g34))

assert(torch.allclose(gamma, gamma1+gamma2-2*gamma3))

# global check
assert(torch.allclose(out, alpha+gamma-2*beta))


assert(torch.allclose(out, alpha1 - 2 * b11 + 2 *
                      b21 + 2*b24 + g11 + gamma2 - 2*g31 - 2*g34))


X = T@invA@Bbar - T@invA@B@invA@Abar
gx = trace(X@X)

Y = invA@B@invA@Bbar@T
gy = trace(Y)

Z = invA@B@invA@B@invA@Abar@T
gz = trace(Z)

assert(torch.allclose(out, alpha1 + gx + 2*(gy-gz)))


M = invA@Bbar - invA@B@invA@Abar
gm = trace(T@M@T@M)

P = T@invA@B@M
gp = trace(P)

print(out, alpha1 + gm + 2*gp)
assert(torch.allclose(out, alpha1 + gm + 2*gp))


Mtilde = invA@Btilde - invA@B@invA@Atilde

Ptilde_left = torch.zeros(4, 2)
Ptilde_left[1, 0] = 1
Ptilde_left[3, 1] = 1

Ptilde_right = torch.zeros(4, 2)
Ptilde_right[0, 0] = 1
Ptilde_right[2, 1] = 1

AA = inverse(proj(invA@Atilde, Ptilde_left, Ptilde_right))
Mproj = proj(Mtilde, Ptilde_left, Ptilde_right)

AAM = AA@Mproj

gm_ = trace(AAM@AAM)


ABM = proj(invA@B@Mtilde, Ptilde_left, Ptilde_right)

gp_ = trace(AA@ABM)

print(out)
print(alpha1+gm_+2*gp_)
