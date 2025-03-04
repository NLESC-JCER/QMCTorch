import dgl
import torch
from mendeleev import element


def ElecNucGraph(natoms:int, atom_types:list, atomic_features:list, nelec:int, nup:int) -> dgl.DGLGraph:
    """Create the elec-nuc graph

    Args:
        nelec (int): total number of electrons
        nup (int): numpber of spin up electrons

    Returns:
        [dgl.DGLGraph]: DGL graph
    """
    edges = get_elec_nuc_edges(natoms, nelec)
    graph = dgl.graph(edges)
    graph.ndata["node_types"] = get_elec_nuc_ndata(
        natoms, atom_types, atomic_features, nelec, nup
    )
    return graph


def get_elec_nuc_edges(natoms: int, nelec: int) -> tuple:
    """Compute the edge index of the electron-nuclei graph."""
    en_edges = ([], [])
    for i in range(natoms):
        for j in range(nelec):
            en_edges[0].append(i)
            en_edges[1].append(natoms + j)

            en_edges[0].append(natoms + j)
            en_edges[1].append(i)

    # for i in range(natoms-1):
    #     for j in range(i+1, natoms):
    #         en_edges[0].append(i)
    #         en_edges[1].append(j)
    return en_edges


def get_elec_nuc_ndata(natoms: int, atom_types: list, atomic_features: list, nelec: int, nup: int) -> torch.Tensor:
    """Compute the node data of the elec-elec graph"""

    en_ndata = []
    embed_number = 0
    atom_dict = {}

    for i in range(natoms):
        if atom_types[i] not in atom_dict:
            atom_dict[atom_types[i]] = embed_number
            en_ndata.append(embed_number)
            embed_number += 1
        else:
            en_ndata.append(atom_dict[atom_types[i]])

        # feat = get_atomic_features(atom_types[i], atomic_features)
        # feat.append(0)  # spin
        # en_ndata.append(feat)

    for i in range(nelec):
        # feat = get_atomic_features(None, atomic_features)
        if i < nup:
            en_ndata.append(embed_number)
        else:
            en_ndata.append(embed_number + 1)

    return torch.LongTensor(en_ndata)


def get_atomic_features(atom_type: list, atomic_features: list) -> list:
    """Get the atomic features requested."""
    if atom_type is not None:
        data = element(atom_type)
        feat = [getattr(data, feat) for feat in atomic_features]
    else:
        feat = []
        for atf in atomic_features:
            if atf == "atomic_number":
                feat.append(-1)
            else:
                feat.append(0)
    return feat
