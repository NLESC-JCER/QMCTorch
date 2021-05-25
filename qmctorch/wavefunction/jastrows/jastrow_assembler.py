

from torch import nn
from .elec_elec.kernels.pade_jastrow_kernel import PadeJastrowKernel
from .elec_elec.jastrow_factor_electron_electron import JastrowFactorElectronElectron
from .elec_nuclei.jastrow_factor_electron_nuclei import JastrowFactorElectronNuclei
from .elec_elec_nuclei.jastrow_factor_electron_electron_nuclei import JastrowFactorElectronElectronNuclei


class UnityTerm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pos, derivative=0, sum_grad=True):
        if derivative == 0:
            return 1.
        elif derivative == 1:
            return 0.
        elif derivative == 2:
            return 0.
        elif derivative == [0, 1, 2]:
            return [1., 0., 0.]


class JastrowFactorCombinedTerms(nn.Module):

    def __init__(self, nup, ndown, atomic_pos,
                 elec_elec_kernel=PadeJastrowKernel,
                 elec_elec_kernel_kwargs={},
                 elec_nuc_kernel=None,
                 elec_nuc_kernel_kwargs={},
                 elec_elec_nuc_kernel=None,
                 elec_elec_nuc_kernel_kwargs={},
                 cuda=False):
        """[summary]

        Args:
            nup ([type]): [description]
            ndown ([type]): [description]
            atomic_pos ([type]): [description]
            elec_elec_kernel ([type], optional): [description]. Defaults to PadeJastrowKernel.
            elec_elec_kernel_kwargs (dict, optional): [description]. Defaults to {}.
            elec_nuc_kernel ([type], optional): [description]. Defaults to None.
            elec_nuc_kernel_kwargs (dict, optional): [description]. Defaults to {}.
            elec_elec_nuc_kernel ([type], optional): [description]. Defaults to None.
            elec_elec_nuc_kernel_kwargs (dict, optional): [description]. Defaults to {}.
            cuda (bool, optional): [description]. Defaults to False.
        """

        super().__init__()
        self.nup = nup
        self.ndown = ndown
        self.cuda = cuda

        if elec_elec_kernel is None:
            self.elec_elec = UnityTerm()
        else:
            self.elec_elec = JastrowFactorElectronElectron(nup, ndown,
                                                           elec_elec_kernel,
                                                           elec_elec_kernel_kwargs,
                                                           cuda=cuda)

        if elec_nuc_kernel is None:
            self.elec_nuc = UnityTerm()
        else:
            self.elec_nuc = JastrowFactorElectronNuclei(nup, ndown,
                                                        atomic_pos,
                                                        elec_nuc_kernel,
                                                        elec_nuc_kernel_kwargs,
                                                        cuda=cuda)

        if elec_elec_nuc_kernel is None:
            self.elec_elec_nuc = UnityTerm()
        else:
            self.elec_elec_nuc = JastrowFactorElectronElectronNuclei(nup, ndown,
                                                                     atomic_pos,
                                                                     elec_elec_nuc_kernel,
                                                                     elec_elec_nuc_kernel_kwargs,
                                                                     cuda=cuda)

    def forward(self, pos, derivative=0, sum_grad=True):
        """Compute the Jastrow factors.

        Args:
            pos(torch.tensor): Positions of the electrons
               Size: Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0, 1, 2,).
              Defaults to 0.
            sum_grad(bool, optional): Return the sum_grad(i.e. the sum of
                                                           the derivatives)
              terms. Defaults to True.
                False only for derivative = 1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
              derivative = 0  (Nmo) x Nbatch x 1
               derivative = 1  (Nmo) x Nbatch x Nelec
                  (for sum_grad = True)
                derivative = 1  (Nmo) x Nbatch x Ndim x Nelec
                  (for sum_grad = False)
        """
        if derivative == 0:
            ee = self.elec_elec(pos)
            en = self.elec_nuc(pos)
            een = self.elec_elec_nuc(pos)

            return self.evaluate_product(ee, en, een)

        elif derivative == 1:

            ee = self.elec_elec(pos)
            en = self.elec_nuc(pos)
            een = self.elec_elec_nuc(pos)

            dee = self.elec_elec(pos, derivative=1, sum_grad=sum_grad)
            den = self.elec_nuc(pos, derivative=1, sum_grad=sum_grad)
            deen = self.elec_elec_nuc(
                pos, derivative=1, sum_grad=sum_grad)

            out = self.evaluate_product(dee, en, een)
            out = out + self.evaluate_product(ee, den, een)
            out = out + self.evaluate_product(ee, en, deen)

            return out

        elif derivative == 2:

            ee, dee, d2ee = self.elec_elec(
                pos, derivative=[0, 1, 2], sum_grad=False)

            en, den, d2en = self.elec_nuc(
                pos, derivative=[0, 1, 2], sum_grad=False)

            een, deen, d2een = self.elec_elec_nuc(
                pos, derivative=[0, 1, 2], sum_grad=False)

            out = self.evaluate_product(d2ee, en, een)
            out = out + self.evaluate_product(ee, d2en, een)
            out = out + self.evaluate_product(ee, en, d2een)

            out = out + 2 * self.evaluate_product(dee, den, een)
            out = out + 2 * self.evaluate_product(dee, en, deen)
            out = out + 2 * self.evaluate_product(ee, den, deen)

            return out

    @staticmethod
    def evaluate_product(a, b, c):
        """Computes the product of a * b * c.
            Returns 0 if either a b or c are null

        Args:
            a(torch.tensor or float): term of the product
            b(torch.tensor or float): term of the product
            c(torch.tensor or float): term of the product
        """

        factor = 1.
        mat = []
        for term in [a, b, c]:
            if isinstance(term, float):
                factor *= term
            else:
                mat.append(term)
        if factor == 0.:
            return 0.
        else:
            out = mat[0]
            for m in mat[1:]:
                out = out * m

        return out
