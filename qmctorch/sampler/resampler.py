from types import SimpleNamespace
from copy import deepcopy


class Resampler(object):

    def __init__(self, sampler, mode='update', resample_every=1, nstep_update=25):

        self.original_sampler = sampler
        self.sampler = deepcopy(sampler)
        self.configure(mode, resample_every, nstep_update)

    def configure(self, mode, resample_every, nstep_update):
        """Configure the resampling

        Args:
            mode (str, optional): method to resample : 'full', 'update', 'never' 
                                  Defaults to 'update'.
            resample_every (int, optional): Number of optimization steps between resampling
                                 Defaults to 1.
            nstep_update (int, optional): Number of MC steps in update mode. 
                                          Defaults to 25.
        """

        self.options = SimpleNamespace()
        valid_mode = ['never', 'full', 'update']
        if mode not in valid_mode:
            raise ValueError(
                mode, 'not a valid update method : ', valid_mode)

        self.options.mode = mode
        self.options.resample_every = resample_every
        self.options.nstep_update = nstep_update

        if mode == 'update':
            self.sampler.ntherm = nstep_update
            self.sampler.nstep = self.sampler.get_number_steps()

    def add_walkers(self, pdf, nwalkers):
        """Add walkers to the sampling."""

        # make a copy of the original sampler
        sampler_copy = deepcopy(self.original_sampler)

        # change the sampler settings
        sampler_copy.nwalkers = nwalkers
        sampler_copy.walkers.nwalkers = nwalkers

        # sample
        pos = sampler_copy(pdf)

        # change the sampler
        self.original_sampler.nwalkers += nwalkers
        self.original_sampler.walkers.nwalkers += nwalkers
        self.original_sampler.nsample = self.original_sampler.get_sampling_size()
        self.original_sampler.walkers.pos = torch.cat(
            (self.original_sampler.walkers.pos, sampler_copy.walkers.pos))

        # change the resampler
        self.sampler.nwalkers += nwalkers
        self.sampler.walkers.nwalkers += nwalkers
        self.sampler.nsample = self.sampler.get_sampling_size()

        # resample
        pos = self(pdf, 0)

        # update the dataloader
        # self.dataset.data = pos
        # if batchsize is None:
        #     batchsize = self.sampler.nsample
        # self.dataloader = DataLoader(
        #     self.dataset, batch_size=batchsize)

        return pos

    def __call__(self, pdf, n, pos):
        """Resample the wave function

        Args:
            n (int): current epoch value
            pos (torch.tensor): positions of the walkers

        Returns:
            (torch.tensor): new positions of the walkers
        """

        if self.options.mode != 'never':

            # resample the data
            if (n % self.options.resample_every == 0):

                # make a copy of the pos if we update
                if self.options.mode == 'update':
                    pos = pos[-self.sampler.walkers.nwalkers:
                              ].clone().detach().to(self.sampler.device)

                # start from scratch otherwise
                else:
                    pos = None

                # sample and update the dataset
                pos = self.sampler(pdf, pos=pos, with_tqdm=True)

        return pos
