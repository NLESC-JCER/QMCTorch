import torch
import numpy as np
from types import SimpleNamespace
from .. import log
from ..utils import dump_to_hdf5, add_group_attr


class Observable(object):

    def __init__(self, obs_name, wf):
        """Configure the observable we want to track

        Args:
            obs_name (list str): names of the observable
            wf (wave function instance): instance of a wave function
        """

        # make sure it's a list
        if not isinstance(obs_name, list):
            obs_name = list(obs_name)

        # check
        self.check(obs_name, wf)

        # configure
        self.configure(obs_name, wf)

    def configure(self, obs_name, wf):
        """Configure the observables.

        Args:
            obs_name (list str): names of the observable
            wf (wave function instance): instance of a wave function
        """

       # add the energy of the sytem
        if 'energy' not in obs_name:
            obs_name += ['energy']

        for k in obs_name:

            if k == 'parameters':
                for key, p in zip(wf.state_dict().keys(),
                                  wf.parameters()):
                    if p.requires_grad:
                        self.__setattr__(key, [])

            elif k == 'gradients':
                for key, p in zip(wf.state_dict().keys(),
                                  wf.parameters()):
                    if p.requires_grad:
                        self.__setattr__(key+'.grad', [])

            else:
                self.__setattr__(k, [])

        self.models = SimpleNamespace()

    def check(self, obs_name, wf):
        """Check that all obs name are valid

        Args:
            obs_name (list str): names of the observable
            wf (wave function instance): instance of a wave function

        Raises:
            ValueError: [description]
        """
        # sanity check
        valid_obs_name = ['energy', 'local_energy',
                          'parameters', 'gradients']
        for name in obs_name:
            if name in valid_obs_name:
                continue

            elif hasattr(wf, name):
                continue
            else:
                log.info(
                    '   Error : Observable %s not recognized' % name)
                log.info('         : Possible observable')
                for n in valid_obs_name:
                    log.info('         :  - %s' % n)
                log.info(
                    '         :  - or any method of the wave function')
                raise ValueError('Observable not recognized')

    def store(self, wf, pos, local_energy=None, ibatch=None, **kwargs):
        """Store the values of the observable

        Args:
            wf(WaveFunction): instance of the wave function
            pos(torch.tensor): current position of the walkers
            local_energy(torch.tensor, optional): precalculated values of the local energies.
                                                   Defaults to None.
            ibatch(int, optional): index of the current minibatch. Defaults to None.
        """

        if wf.cuda and pos.device.type == 'cpu':
            pos = pos.to(self.device)

        for obs in self.__dict__.keys():

            # store the energy
            if obs == 'energy' and local_energy is not None:
                data = local_energy.cpu().detach().numpy()
                if (ibatch is None) or (ibatch == 0):
                    self.energy.append(np.mean(data))
                else:
                    self.energy[-1] *= ibatch/(ibatch+1)
                    self.energy[-1] += np.mean(data)/(ibatch+1)

            # store local energy
            elif obs == 'local_energy' and local_energy is not None:
                data = local_energy.cpu().detach().numpy()
                if (ibatch is None) or (ibatch == 0):
                    self.local_energy.append(data)
                else:
                    self.local_energy[-1] = np.append(
                        self.local_energy[-1], data)

            # store variational parameter
            elif obs in wf.state_dict():
                layer, param = obs.split('.')
                p = wf.__getattr__(layer).__getattr__(param)
                self.__getattribute__(obs).append(
                    p.data.cpu().numpy())

                if obs+'.grad' in self.__dict__.keys():
                    if p.grad is not None:
                        self.__getattribute__(obs +
                                              '.grad').append(p.grad.cpu().numpy())
                    else:
                        self.__getattribute__(obs +
                                              '.grad').append(torch.zeros_like(p.data).cpu().numpy())

            # store any other defined method
            elif hasattr(wf, obs):
                func = wf.__getattribute__(obs)
                data = func(pos)
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach().numpy()
                self.__getattribute__(obs).append(data)

    def store_model(self, name, state_dict):
        """Store a model in the observable dict

        Args:
            name(str): name of the model
            state_dict(state_dict): state dictionary of the model
        """
        self.models.__setattr__(name, dict(state_dict))

    def print(self, cumulative_loss, verbose=False):
        """Print the observalbe to csreen

        Args:
            cumulative_loss(float): current loss value
            verbose(bool, optional): print all the observables. Defaults to False
        """

        for k in self.__dict__.keys():

            if k == 'local_energy':

                eloc = self.local_energy[-1]
                e = np.mean(eloc)
                v = np.var(eloc)
                err = np.sqrt(v / len(eloc))
                log.options(style='percent').info(
                    '  energy   : %f +/- %f' % (e, err))
                log.options(style='percent').info(
                    '  variance : %f' % np.sqrt(v))

            elif verbose:
                log.options(style='percent').info(
                    k + ' : ', self.__getattribute__(k)[-1])
                log.options(style='percent').info(
                    'loss %f' % (cumulative_loss))

    def save(self, hdf5grp, hdf5file, attr_type=None):
        """Save the data in the hdf5 file

        Args:
            hdf5grp(str): name of group in the hdf5 file
            hdf5file(str): name of the hdf5file
            attr_type(str, optional): type of simulation
        """

        hdf5grp = dump_to_hdf5(
            self, hdf5file, hdf5grp)

        if attr_type is not None:
            add_group_attr(hdf5file, hdf5grp, {'type': attr_type})
