import numpy as np
from pyCHAMP.sampler.sampler_base import SAMPLER_BASE
from pyCHAMP.sampler.walkers import WALKERS

class DIFFUSION(SAMPLER_BASE):

    def __init__(self, nwalkers=1000, nstep=1000, nelec=1, ndim=3, 
                 step_size = 0.1, domain = {'min':-2,'max':2},
                 move='all'):

        ''' Diffusion  SAMPLER
        Args:
            f (func) : function to sample
            nstep (int) : number of mc step
            nwalkers (int) : number of walkers
            eps (float) : size of the mc step
            boudnary (float) : boudnary of the space
        '''

        SAMPLER_BASE.__init__(self,nwalkers,nstep,nelec,ndim,step_size,domain,move)


    def set_wf(self,func):
        self.wf = func

    def set_drift_func(self,func):
        self.drift_func = func

    def set_energy_func(self,func):
        self.energy_func = func

    def set_initial_guess(self,pos):
        self.init_pos = pos

    def generate(self):

        self.walkers.initialize(pos=self.init_pos)

        Ex = self.energy_func(self.walkers.pos)
        E = np.mean(Ex)

        for istep in range(self.nstep):

            # drift force
            F = self.drift_func(self.walkers.pos)

            old_pos = self.walkers.pos
            old_wf = self.wf(old_pos)

            for ielec in range(self.walkers.nelec):

                # proposed move
                new_pos = self.walkers.move_dmc(self.step_size,F,ielec)

                # wf values
                new_wf = self.wf(new_pos)    
                
                # fixed node
                FN = np.sign(new_wf) * np.sign(old_wf) 

                #weight
                W = self.weight( new_pos,new_wf,old_pos,old_wf,E )
                W *= FN

                # accept the moves
                index = self._accept(W)
                self.walkers.pos[index,:] = new_pos[index,:]

            # new local energies
            Exn = self.energy_func(self.walkers.pos)

            # update the walkers
            self.walkers_death_birth(Exn,Ex,E)

            # update total energy
            Ex = self.energy_func(self.walkers.pos)
            E = np.mean(Ex)

        return self.walkers.pos

    def weight(self,new_pos,new_wf,old_pos,old_wf,E):
        return (new_wf**2 * self.green(new_pos,old_pos,E)) / ( (old_wf**2 * self.green(old_pos,new_pos,E)) + 1E-6 ) 
        


    def green(self,rn,r,E):
        en = self.energy_func(rn)
        e = self.energy_func(r)
        Gb = np.exp(-0.5*(en+e-2*E)*self.step_size)
        Gd = (2*np.pi*self.step_size)**(-3/2*self.nwalkers) *np.exp( -0.5*(r-rn-0.5*self.step_size*self.drift_func(rn)/self.step_size ) )
        return Gb*Gd


    def _accept(self,df):
        ones = np.ones((self.walkers.nwalkers,1))
        P = np.minimum(ones,df)
        tau = np.random.rand(self.walkers.nwalkers,1)
        return (P-tau>=0).reshape(-1)



    def walkers_death_birth(self,Exn,Ex,E):

        # number of each walker
        N = np.exp(-0.5*self.step_size*(Ex+Exn-2*E)).astype('int')

        # kill the walkers
        index_dead = np.where(N==0)[0]
        num_dead = len(index_dead)
        self.walkers.status[index_dead] = 0

        # multiply walkers
        index_birth = np.where(N>1)[0]
        num_new = len(index_birth)
        new_pos = self.walkers.pos[index_birth,:]

        # track number of walkers
        self.walkers.nwalkers += num_new - num_dead
        print(self.walkers.nwalkers)
        
        # remove dead walkers
        self.walkers.status = np.delete(self.walkers.status,index_dead,axis=0)
        self.walkers.pos = np.delete(self.walkers.pos,index_dead,axis=0)
        
        #add copies
        self.walkers.status = np.concatenate((self.walkers.status,np.ones((num_new,1))))
        self.walkers.pos = np.vstack((self.walkers.pos,new_pos))
        













