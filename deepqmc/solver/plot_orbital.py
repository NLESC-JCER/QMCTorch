from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from mendeleev import element
from mayavi import mlab
from deepqmc.solver.torch_utils import Loss, OrthoReg


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16)/255
                 for i in range(0, hlen, hlen//3))


def plot_molecule(solver, pos=None, loss='variance', alpha=0.025):

    # get sampling points
    if pos is None:
        pos = Variable(solver.sample())
    pos.requires_grad = True

    # get loss
    if loss is not None:
        loss = Loss(solver.wf, method=loss)
        ortho_loss = OrthoReg()
        loss_val = loss(pos)
        if solver.wf.mo.weight.requires_grad:
            loss_val += ortho_loss(solver.wf.mo.weight)
        loss_val.backward()

    # plot
    plt.style.use('dark_background')
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_axis_off()

    # plot electrons
    d = pos.detach().numpy()
    for ielec in range(solver.wf.nelec):
        ax.scatter(d[:, 3*ielec], d[:, 3*ielec+1], d[:, 3*ielec+2],
                   alpha=alpha)

    # plot atoms
    for iat in range(solver.wf.natom):

        at = solver.wf.atoms[iat]
        atdata = element(at)

        r = (atdata.vdw_radius)
        col = atdata.jmol_color

        x, y, z = solver.wf.ao.atom_coords.data[iat, :].numpy()
        ax.scatter(x, y, z, s=r, color=col)

        if loss is not None:
            if (solver.wf.ao.atom_coords.requires_grad is True):
                u, v, w = solver.wf.ao.atom_coords.grad.data[iat, :].numpy()
                ax.quiver(x, y, z, -u, -v, -w, color='grey', length=1.,
                          normalize=True, pivot='middle')

    plt.show()


def plot_molecule_mayavi(solver, pos=None, loss=None, alpha=0.05):

    # get sampling points
    if pos is None:
        pos = Variable(solver.sample())
    pos.requires_grad = True

    # get loss
    if loss is not None:
        loss = Loss(solver.wf, method=loss)
        ortho_loss = OrthoReg()
        loss_val = loss(pos)
        if solver.wf.mo.weight.requires_grad:
            loss_val += ortho_loss(solver.wf.mo.weight)
        loss_val.backward()

    # plot
    mlab.clf()

    # plot electrons
    d = pos.detach().numpy()
    for ielec in range(solver.wf.nelec):
        mlab.points3d(d[:, 3*ielec], d[:, 3*ielec+1], d[:, 3*ielec+2],
                      scale_factor=0.1,
                      resolution=5,
                      color=cm.jet(int(256*(ielec/solver.wf.nelec)))[:-1],
                      opacity=alpha)

    # plot atoms
    for iat in range(solver.wf.natom):

        at = solver.wf.atoms[iat]
        atdata = element(at)

        r = (atdata.vdw_radius)/200
        col = atdata.jmol_color

        x, y, z = solver.wf.ao.atom_coords.data[iat, :].numpy()
        mlab.points3d(x, y, z,
                      mode='sphere',
                      scale_factor=r,
                      resolution=20,
                      color=hex_to_rgb(col))

        if loss is not None:
            if solver.wf.ao.atom_coords.requires_grad is True:
                u, v, w = solver.wf.ao.atom_coords.grad.data[iat, :].numpy()
                mlab.quiver3d(x, y, z, -u, -v, -w,
                              color=(1, 0, 0),
                              line_width=3,
                              scale_factor=5,
                              mode='arrow')

    # plot the bonds
    for bindex in solver.wf.bonds:
        iat1, iat2 = bindex
        xyz1 = solver.wf.ao.atom_coords.data[iat1, :].numpy()
        xyz2 = solver.wf.ao.atom_coords.data[iat2, :].numpy()
        pos = np.vstack((xyz1, xyz2))
        mlab.plot3d(pos[:, 0], pos[:, 1], pos[:, 2], color=(0.5, 0.5, 0.5),
                    tube_radius=0.05)

    # mlab.view(132,54,45,[21,20,21.5])
    mlab.show()
