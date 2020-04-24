Optimizers
===========================

`QMCTorch` allows to use all the optimizers included in `pytorch` to opmtimize the QMCNet wave function.
The list of optimizers can be found here :  https://pytorch.org/docs/stable/optim.html 

For example to use the ADAM optimizer with different learning rate for each layer of the QMCNet wave funciton, one can simply define :

>>> from torch import optim
>>> lr_dict = [{'params': wf.jastrow.parameters(), 'lr': 1E-3},
>>>        {'params': wf.ao.parameters(), 'lr': 1E-6},
>>>        {'params': wf.mo.parameters(), 'lr': 1E-3},
>>>        {'params': wf.fc.parameters(), 'lr': 1E-3}]
>>> opt = optim.Adam(lr_dict, lr=1E-3)

Scheduler
==============================

Similarly QMCTorch allows to use scheduler to gradually decrease the learning rate during the optimization.
There as well all the scheduler of pytorch can be used : https://pytorch.org/docs/stable/optim.html
For example a simple scheudler that decrease the learning rate every number of epoch is simply defined as :

>>> from torch import optim
>>> scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.90)



