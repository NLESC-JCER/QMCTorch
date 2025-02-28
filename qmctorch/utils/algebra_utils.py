import torch
import numpy as np
from typing import List
from scipy.special import factorial2 as f2


def btrace(M: torch.Tensor) -> torch.Tensor:
    """Computes the trace of batched matrices

    Args:
        M: matrices of size (Na, Nb, ... Nx, N, N)

    Returns:
        trace of matrices (Na, Nb, ... Nx)
    """
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def bproj(M: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project batched matrices using P^T M P

    Args:
        M (torch.Tensor): Batched matrices of size (..., N, M)
        P (torch.Tensor): Projectors of size (..., N, M)

    Returns:
        torch.Tensor: Projected matrices
    """
    return P.transpose(-1, -2) @ M @ P


def bdet2(M: torch.Tensor) -> torch.Tensor:
    """Computes the determinant of batched 2x2 matrices

    Args:
        M (torch.tensor): input matrices

    Returns:
        torch.tensor: determinants of the matrices
    """

    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]


def double_factorial(input: List) -> np.ndarray:
    """Computes the double factorial of an array of int

    Args:
        input (List): input numbers

    Returns:
        List: values of the double factorial
    """
    output = f2(input)
    return np.array([1 if o == 0 else o for o in output])


class BatchDeterminant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # LUP decompose the matrices
        inp_lu, pivots = input.lu()
        _, _, inpu = torch.lu_unpack(inp_lu, pivots)

        # get the number of permuations
        s = (
            (pivots != torch.as_tensor(range(1, input.shape[1] + 1)).int())
            .sum(1)
            .type(torch.get_default_dtype())
        )

        # get the prod of the diag of U
        d = torch.diagonal(inpu, dim1=-2, dim2=-1).prod(1)

        # assemble
        det = (-1) ** s * d
        ctx.save_for_backward(input, det)

        return det

    @staticmethod
    def backward(ctx, grad_output):
        """using jaobi's formula
            d det(A) / d A_{ij} = adj^T(A)_{ij}
        using the adjunct formula
            d det(A) / d A_{ij} = ( (det(A) A^{-1})^T )_{ij}
        """
        input, det = ctx.saved_tensors
        return (grad_output * det).view(-1, 1, 1) * torch.inverse(input).transpose(1, 2)
