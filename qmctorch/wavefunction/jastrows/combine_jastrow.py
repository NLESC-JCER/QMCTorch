from typing import List, Union
import torch
from torch import nn
from functools import reduce


class CombineJastrow(nn.Module):
    def __init__(self, jastrow: List[nn.Module]) -> None:
        """Combine several Jastrow Factors

        Args:
            jastrow (list) : list of jastrow factor
        """

        super().__init__()

        self.jastrow_terms = nn.ModuleList()
        for j in jastrow:
            self.jastrow_terms.append(j)

        self.requires_autograd = True

        self.nterms = len(self.jastrow_terms)

    def __repr__(self):
        """representation of the jastrow factor"""
        out = []
        for term in self.jastrow_terms:
            out.append(term.__repr__())

        return " + ".join(out)

    def forward(
        self, pos: torch.Tensor, derivative: int = 0, sum_grad: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute the Jastrow factors and/or its derivatives

        Args:
            pos(torch.tensor): Positions of the electrons
               Size: Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0, 1, 2,).
              Defaults to 0.
            sum_grad(bool, optional): Return the sum_grad(i.e. the sum of
                                                           the derivatives)

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
              derivative = 0  (Nmo) x Nbatch x 1
              derivative = 1  (Nmo) x Nbatch x Nelec
                  (for sum_grad = True)
              derivative = 1  (Nmo) x Nbatch x Ndim x Nelec
                  (for sum_grad = False)
        """
        if derivative == 0:
            jast_vals = [term(pos) for term in self.jastrow_terms]
            return self.get_combined_values(jast_vals)

        elif derivative == 1:
            if sum_grad:
                jast_vals = [term(pos) for term in self.jastrow_terms]
            else:
                jast_vals = [term(pos).unsqueeze(-1) for term in self.jastrow_terms]
            djast_vals = [
                term(pos, derivative=1, sum_grad=sum_grad)
                for term in self.jastrow_terms
            ]

            return self.get_derivative_combined_values(jast_vals, djast_vals)

        elif derivative == 2:
            jast_vals = [term(pos) for term in self.jastrow_terms]
            djast_vals = [
                term(pos, derivative=1, sum_grad=False) for term in self.jastrow_terms
            ]
            d2jast_vals = [term(pos, derivative=2) for term in self.jastrow_terms]
            return self.get_second_derivative_combined_values(
                jast_vals, djast_vals, d2jast_vals
            )

        elif derivative == [0, 1, 2]:
            jast_vals = [term(pos) for term in self.jastrow_terms]
            djast_vals = [
                term(pos, derivative=1, sum_grad=False) for term in self.jastrow_terms
            ]
            d2jast_vals = [term(pos, derivative=2) for term in self.jastrow_terms]

            # combine the jastrow terms
            out_jast = self.get_combined_values(jast_vals)

            # combine the second derivative
            out_d2jast = self.get_second_derivative_combined_values(
                jast_vals, djast_vals, d2jast_vals
            )

            # unsqueeze the jast terms to be compatible with the
            # derivative
            jast_vals = [j.unsqueeze(-1) for j in jast_vals]

            # combine the derivative
            out_djast = self.get_derivative_combined_values(jast_vals, djast_vals)

            return (out_jast, out_djast, out_d2jast)

        else:
            raise ValueError("derivative not understood")

    @staticmethod
    def get_combined_values(jast_vals: List[torch.Tensor]) -> torch.Tensor:
        """Compute the product of all terms in jast_vals."""
        if len(jast_vals) == 1:
            return jast_vals[0]
        else:
            return reduce(lambda x, y: x * y, jast_vals)

    @staticmethod
    def get_derivative_combined_values(
        jast_vals: List[torch.Tensor], djast_vals: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the derivative of the product of Jastrow terms.

        This function calculates the first derivative of a product of Jastrow
        factors with respect to their input variables. The computation is based
        on the formula:

        .. math::
            J = A * B * C
            \\frac{d J}{dx} = \\frac{d A}{dx} B C + A \\frac{d B}{dx} C + A B \\frac{d C}{dx}

        Args:
            jast_vals (List[torch.Tensor]): List of Jastrow values.
            djast_vals (List[torch.Tensor]): List of first derivatives of Jastrow values.

        Returns:
            torch.Tensor: The derivative of the product of Jastrow terms.
        """
        if len(djast_vals) == 1:
            return djast_vals[0]
        else:
            out = 0.0
            nterms = len(jast_vals)
            for i in range(nterms):
                tmp = jast_vals.copy()
                tmp[i] = djast_vals[i]
                out += reduce(lambda x, y: x * y, tmp)
            return out

    @staticmethod
    def get_second_derivative_combined_values(
        jast_vals: List[torch.Tensor],
        djast_vals: List[torch.Tensor],
        d2jast_vals: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the second derivative of the product of Jastrow terms.

        This function calculates the second derivative of a product of Jastrow
        factors with respect to their input variables. The computation is based
        on the formula:

        .. math::
            J = A * B * C
            \\frac{d^2 J}{dx^2} = \\frac{d^2 A}{dx^2} B C + A \\frac{d^2 B}{dx^2} C + A B \\frac{d^2 C}{dx^2} \\
                            + 2( \\frac{d A}{dx} \\frac{dB}{dx} C + \\frac{d A}{dx} B \\frac{dC}{dx} + A \\frac{d B}{dx} \\frac{dC}{dx} )

        Args:
            jast_vals (List[torch.Tensor]): List of Jastrow values.
            djast_vals (List[torch.Tensor]): List of first derivatives of Jastrow values.
            d2jast_vals (List[torch.Tensor]): List of second derivatives of Jastrow values.

        Returns:
            torch.Tensor: The combined second derivative of the Jastrow factors.
        """

        if len(d2jast_vals) == 1:
            return d2jast_vals[0]

        # otherwise
        out = 0.0
        nterms = len(jast_vals)
        for i in range(nterms):
            # d2a * b * c
            tmp = jast_vals.copy()
            tmp[i] = d2jast_vals[i]
            out = out + reduce(lambda x, y: x * y, tmp)

        for i in range(nterms - 1):
            for j in range(i + 1, nterms):
                # da * db * c
                tmp = jast_vals.copy()
                tmp = [j.unsqueeze(-1) for j in tmp]
                tmp[i] = djast_vals[i]
                tmp[j] = djast_vals[j]

                out = out + (2.0 * reduce(lambda x, y: x * y, tmp)).sum(1)

        return out
