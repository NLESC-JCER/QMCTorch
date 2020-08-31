from .solver_orbital import SolverOrbital
from .solver_base_horovod import SolverBaseHorovod


class SolverOrbitalHorovod(SolverBaseHorovod, SolverOrbital):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None, output=None, rank=0):

        super().__init__(wf, sampler, optimizer, scheduler, output, rank)

        # set which parameter to optimize
        self.configure_parameters(freeze=None)

        # how to compute the grad of the parameters
        self.configure_gradients('manual')

        # loss to use
        self.configure_loss(loss='energy', clip=False, ortho_mo=False)
