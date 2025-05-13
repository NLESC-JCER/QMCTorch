from abc import ABC
import torch


def planar_symmetry(
    pos: torch.tensor, plane: str, nelec: int, ndim: int, inplace=False
):
    """
    Apply a planar symmetry operation to a set of positions.

    Args:
        pos (torch.tensor): The input tensor representing positions,
            expected shape is (N, ndim * nelec).
        plane (str): The plane of symmetry, can be 'xy', 'xz', or 'yz'.
        nelec (int): Number of electrons (or particles).
        ndim (int): Number of dimensions per electron.
        inplace (bool, optional): If True, modify the input tensor in place.
            Defaults to False.

    Returns:
        torch.tensor: A tensor with the planar symmetry operation applied.
    """
    if inplace:
        out = pos
    else:
        out = torch.clone(pos)

    if not isinstance(plane, list):
        plane = [plane]

    for p in plane:
        offset = {"xy": 2, "xz": 1, "yz": 0}[p]
        out[:, [ndim * ielec + offset for ielec in range(nelec)]] *= -1.0
    return out


class BaseSymmetry(ABC):
    def __init__(self, label: str):
        self.label = label
        self.nelec = None
        self.ndim = 3

    def __call__(self, pos: torch.tensor) -> torch.tensor:
        raise NotImplementedError


class C1(BaseSymmetry):
    def __init__(self):
        """
        Initialize the C1 symmetry (No symmetry)

        Parameters
        ----------
        label : str
            The name of the symmetry.

        """
        super().__init__("C1")

    def __call__(self, pos: torch.tensor) -> torch.tensor:
        """
        Apply the symmetry to a given position.

        Parameters
        ----------
        pos : torch.tensor
            The positions of the walkers. The shape of the tensor is (Nbatch, Nelec x Ndim).

        Returns
        -------
        torch.tensor
            The positions with the symmetry applied.
        """
        return pos


class Cinfv(BaseSymmetry):
    def __init__(self, axis: str):
        """
        Initialize the Cinfv symmetry (Infinite axis of symmetry).

        Parameters
        ----------
        label : str
            The name of the symmetry.
        axis : str
            The axis of symmetry. Can be 'x', 'y', or 'z'.

        """
        super().__init__("Cinfv")
        if axis not in ["x", "y", "z"]:
            raise ValueError(f"Axis {axis} is not valid. Must be 'x', 'y', or 'z'.")
        self.axis = axis
        self.symmetry_planes = {
            "x": ["xy", "xz"],
            "y": ["xy", "yz"],
            "z": ["xz", "yz"],
        }[self.axis]

    def __call__(self, pos: torch.tensor) -> torch.tensor:
        """
        Apply the symmetry to a given position.

        Parameters
        ----------
        pos : torch.tensor
            The positions of the walkers. The shape of the tensor is (Nbatch, Nelec x Ndim).

        Returns
        -------
        torch.tensor
            The positions with the symmetry applied.
        """
        if self.nelec is None:
            self.nelec = pos.shape[1] // self.ndim

        symmetry_pos = []
        symmetry_pos.append(pos)
        for plane in self.symmetry_planes:
            symmetry_pos.append(
                planar_symmetry(pos, plane, self.nelec, self.ndim, inplace=False)
            )
        symmetry_pos.append(
            planar_symmetry(
                pos, self.symmetry_planes, self.nelec, self.ndim, inplace=False
            )
        )
        return torch.cat(symmetry_pos, dim=0).requires_grad_(pos.requires_grad)


class Dinfh(BaseSymmetry):
    def __init__(self, axis: str):
        """
        Initialize the Dinfh symmetry (Infinite dihedral symmetry).

        Parameters
        ----------
        label : str
            The name of the symmetry.
        axis : str
            The axis of symmetry. Can be 'x', 'y', or 'z'.
        """

        super().__init__("Dinfv")
        if axis not in ["x", "y", "z"]:
            raise ValueError(f"Axis {axis} is not valid. Must be 'x', 'y', or 'z'.")
        self.axis = axis
        self.symmetry_planes = {
            "x": ["xy", "xz"],
            "y": ["xy", "yz"],
            "z": ["xz", "yz"],
        }[self.axis]
        self.last_symmetry = {"x": "yz", "y": "xz", "z": "xy"}[self.axis]

    def __call__(self, pos: torch.tensor) -> torch.tensor:
        """
        Apply the symmetry to a given position.

        Parameters
        ----------
        pos : torch.tensor
            The positions of the walkers. The shape of the tensor is (Nbatch, Nelec x Ndim).

        Returns
        -------
        torch.tensor
            The positions with the symmetry applied.
        """
        if self.nelec is None:
            self.nelec = pos.shape[1] // self.ndim
        symmetry_pos = []
        symmetry_pos.append(pos)
        for plane in self.symmetry_planes:
            symmetry_pos.append(
                planar_symmetry(pos, plane, self.nelec, self.ndim, inplace=False)
            )
        symmetry_pos.append(
            planar_symmetry(
                pos, self.symmetry_planes, self.nelec, self.ndim, inplace=False
            )
        )
        symmetry_pos.append(
            planar_symmetry(
                torch.cat(symmetry_pos, dim=0),
                self.last_symmetry,
                self.nelec,
                self.ndim,
                inplace=False,
            )
        )
        return torch.cat(symmetry_pos, dim=0).requires_grad_(pos.requires_grad)
