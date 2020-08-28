
def save_trajectory(fname, atoms, xyz):
    """[summary]

    Args:
        fname (str): filename
        atoms (list): atom names (natoms)
        xyz (torch.tensor): positions (nstep, natoms)
    """
    f = open(fname, 'w')
    natom = len(xyz[0])
    nm2bohr = 1.88973
    for snap in xyz:
        f.write('%d \n\n' % natom)
        for i, pos in enumerate(snap):
            at = atoms[i]
            f.write('%s % 7.5f % 7.5f %7.5f\n' % (at[0],
                                                  pos[0] /
                                                  nm2bohr,
                                                  pos[1] /
                                                  nm2bohr,
                                                  pos[2] / nm2bohr))
        f.write('\n')
    f.close()
