Quantum Monte Carlo: a 1 min introduction
===========================================

Quantum Monte Carlo simulations rely on the variational principle:

.. math::

    E = \frac{\int \Psi^*_\theta(R) \; H \; \Psi_\theta(R) dR}{\int |\Psi_\theta(R)|^2} \geq E_0

where :math:`\Psi_\theta(R)` is the wave function of the system computed for the atomic and electronic positions :math:`R`, 
and with variational parameters :math:`\theta`, :math:`H` is the Hamiltonian of the system given by:

.. math::

    H = -\frac{1}{2}\sum_i \Delta_i + \sum_{i>j} \frac{1}{|r_i-r_j|} - \sum_{i\alpha} \frac{Z_\alpha}{|r_i-R_\alpha|} - \sum_{\alpha>\beta}\frac{Z_\alpha Z_\beta}{|R_\alpha-R_\beta|}

where :math:`\Delta_i` is the Laplacian w.r.t the i-th electron, :math:`r_i` is the position of the i-th electron, :math:`R_\alpha` 
the position of the :math:`\alpha`-th atom and :math:`Z_\alpha` its atomic number. QMC simulations express this integral as:

.. math::

    E = \int \rho(R)E_L(R)dR \geq E_0

with:

.. math::

    \rho(R) = \frac{|\Psi_\theta(R)|^2}{\int |\Psi_\theta(R)|^2 dR}

reprensent the denisty associated with the wave function, and:

.. math::

    E_L(R) = \frac{H\Psi_\theta(R)}{\Psi_\theta(R)}

are the so called local energies of the system. QMC simulation then approximated the total energy as:

.. math::
    E \approx \frac{1}{M}\sum_i^M \frac{H\Psi_\theta(R_i)}{\Psi_\theta(R_i)}

where :math:`R_i` are samples of the density :math:`\rho` for example obtained via Metropolis Hasting sampling. 
QMC simulations rely then on the optimization of the variational parameters of the wave function, :math:`\theta`, to minimize the value
of the total energy of the system.

.. image:: ../pics/qmc.png