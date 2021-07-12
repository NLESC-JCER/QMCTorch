import dgl
import torch
from mendeleev import element


def ElecNucGraph(natoms, atom_types, atomic_features, nelec, nup):
    """Create the elec-nuc graph

    Args:
        nelec (int): total number of electrons
        nup (int): numpber of spin up electrons

    Returns:
        [dgl.DGLGraph]: DGL graph
    """
    edges = get_elec_nuc_edges(natoms, nelec)
    graph = dgl.graph(edges)
    graph.ndata["features"] = get_elec_nuc_ndata(
        natoms, atom_types, atomic_features, nelec, nup)
    return graph


def get_elec_nuc_edges(natoms, nelec):
    """Compute the edge index of the electron-nuclei graph.
    """
    en_edges = ([], [])
    for i in range(natoms):
        for j in range(nelec):
            en_edges[0].append(i)
            en_edges[1].append(natoms+j)

    for i in range(natoms-1):
        for j in range(i+1, natoms):
            en_edges[0].append(i)
            en_edges[1].append(j)
    return en_edges


def get_elec_nuc_ndata(natoms, atom_types, atomic_features,  nelec, nup):
    """Compute the node data of the elec-elec graph
    """

    en_ndata = []
    for i in range(natoms):
        feat = get_atomic_features(atom_types[i], atomic_features)
        feat.append([0])  # spin
        en_ndata.append(feat)

    for i in range(nelec):
        feat = get_atomic_features(None, atomic_features)
        if i < nup:
            feat.append(1)
        else:
            feat.append(-1)
        en_ndata.append(feat)

    return en_ndata


def get_atomic_features(atom_type, atomic_features):
    """Get the atomic features requested.
    """
    if atom_type is not None:
        data = element(atom_type)
        feat = [getattr(data, feat)
                for feat in atomic_features]
    else:
        feat = []
        for atf in atomic_features:
            if atf == 'atomic_number':
                feat.append(-1)
            else:
                feat.append(0)
    return feat
