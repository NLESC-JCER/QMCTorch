from torch import nn


class UnityTerm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pos, derivative=0, jacobian=True):
        if derivative == 0:
            return 1.
        elif derivative == 1:
            return 0.
        elif derivative == 2:
            return 0.
        elif derivative == [0, 1, 2]:
            return [1., 0., 0.]


class JastrowAssembler(nn.Module):

    def __init__(self, nup, ndown, jastrow_terms, cuda=False):
        """Jastrow product

        Args:
            nup ([type]): [description]
            ndown ([type]): [description]
            jastrow_terms ([type]): [description]
            cuda (bool, optional): [description]. Defaults to False.
        """

        super().__init__()

        for k in jastrow_terms.keys():
            if k not in ['elec_nuc', 'elec_elec', 'elec_elec_nuc']:
                raise KeyError("Jastrow key not valid")

        if 'elec_nuc' not in jastrow_terms.keys():
            jastrow_terms['elec_nuc'] = UnityTerm()

        if 'elec_elec' not in jastrow_terms.keys():
            jastrow_terms['elec_elec'] = UnityTerm()

        if 'elec_elec_nuc' not in jastrow_terms.keys():
            jastrow_terms['elec_elec_nuc'] = UnityTerm()

        self.elec_nuc = jastrow_terms['elec_nuc']
        self.elec_elec = jastrow_terms['elec_elec']
        self.elec_elec_nuc = jastrow_terms['elec_elec_nuc']

    def forward(self, pos, derivative=0, jacobian=True):
        """Compute the Jastrow factors.

        Args:
            pos(torch.tensor): Positions of the electrons
               Size: Nbatch, Nelec x Ndim
            derivative (int, optional): order of the derivative (0, 1, 2,).
              Defaults to 0.
            jacobian(bool, optional): Return the jacobian(i.e. the sum of
                                                           the derivatives)
              terms. Defaults to True.
                False only for derivative = 1

        Returns:
            torch.tensor: value of the jastrow parameter for all confs
              derivative = 0  (Nmo) x Nbatch x 1
               derivative = 1  (Nmo) x Nbatch x Nelec
                  (for jacobian = True)
                derivative = 1  (Nmo) x Nbatch x Ndim x Nelec
                  (for jacobian = False)
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

            dee = self.elec_elec(pos, derivative=1, jacobian=jacobian)
            den = self.elec_nuc(pos, derivative=1, jacobian=jacobian)
            deen = self.elec_elec_nuc(
                pos, derivative=1, jacobian=jacobian)

            out = self.evaluate_product(dee, en, een)
            out = out + self.evaluate_product(ee, den, een)
            out = out + self.evaluate_product(ee, en, deen)

            return out

        elif derivative == 2:

            ee, dee, d2ee = self.elec_elec(
                pos, derivative=[0, 1, 2], jacobian=False)

            en, den, d2en = self.elec_nuc(
                pos, derivative=[0, 1, 2], jacobian=False)

            een, deen, d2een = self.elec_elec_nuc(
                pos, derivative=[0, 1, 2], jacobian=False)

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
