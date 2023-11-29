import dgl
import torch


def ElecElecGraph(nelec, nup):
    """Create the elec-elec graph

    Args:
        nelec (int): total number of electrons
        nup (int): numpber of spin up electrons

    Returns:
        [dgl.DGLGraph]: DGL graph
    """
    edges = get_elec_elec_edges(nelec)
    graph = dgl.graph(edges)
    graph.ndata["node_types"] = get_elec_elec_ndata(nelec, nup)
    return graph


def get_elec_elec_edges(nelec):
    """Compute the edge index of the electron-electron graph."""
    ee_edges = ([], [])
    for i in range(nelec - 1):
        for j in range(i + 1, nelec):
            ee_edges[0].append(i)
            ee_edges[1].append(j)

            ee_edges[0].append(j)
            ee_edges[1].append(i)

    return ee_edges


def get_elec_elec_ndata(nelec, nup):
    """Compute the node data of the elec-elec graph"""

    ee_ndata = []
    for i in range(nelec):
        if i < nup:
            ee_ndata.append(0)
        else:
            ee_ndata.append(1)

    return torch.LongTensor(ee_ndata)
