import torch


def btrace(M):
    """Computes the trace of batched matrices

    Args:
        M (torch.tensor): matrices of size (Na, Nb, ... Nx, N, N)

    Returns:
        torch.tensor: trace of matrices (Na, Nb, ... Nx)

    Example:
        >>> m = torch.rand(100,5,5)
        >>> tr = btrace(m)
    """
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def bproj(M, P):
    """Project batched marices using P^T M P

    Args:
        M (torch.tensor): batched matrices size (..., N, M)
        P (torch.tensor): Porjectors size (..., N, M)

    Returns:
        torch.tensor: Projected matrices
    """
    return P.transpose(1, 2) @ M @ P


def bdet2(M):
    """Computes the determinant of batched 2x2 matrices

    Args:
        M (torch.tensor): input matrices

    Returns:
        torch.tensor: determinants of the matrices
    """

    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
