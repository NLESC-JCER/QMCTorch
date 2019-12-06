import numpy as np


def regular_mesh_2d(xmin=-2, xmax=2, ymin=-2., ymax=2, nx=5, ny=5):

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    XX, YY = np.meshgrid(x, y)
    points = np.vstack((XX.flatten(), YY.flatten())).T

    return points.tolist()


def regular_mesh_3d(xmin=-2, xmax=2, ymin=-2., ymax=2, zmin=-5, zmax=5, nx=5, ny=5, nz=5):

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)

    XX, YY, ZZ = np.meshgrid(x, y, z)
    points = np.vstack((XX.flatten(), YY.flatten(), ZZ.flatten())).T

    return points.tolist()
