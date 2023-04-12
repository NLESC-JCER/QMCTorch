Quantum Monte Carlo: a 1 min introduction
========================

Quantum Monte Carlo simulations rely on the variational principle:

.. math::

    E = \frac{\int \Psi^*_\theta(R) \; H \; \Psi_theta(R) dR}{\int |\Psi_theta(R)|^2} \geq E_0

where $\Psi_\theta(R)$ is the wave function of the system computed for the atomic and electronic positions $R$, and with variational parameters $\theta$, $H$ is the Hamiltonian of the system given by:

.. math::

    H = -\frac{1}{2}\sum_i \Delta_i + \sum_{i>j} \frac{1}{|r_i-r_j|} - \sum_{i\alpha} \frac{Z_\alpha}{|r_i-R_\alpha|} - \sum_{\alpha>\beta}\frac{Z_\alpha Z_\beta}{|R_\alpha-R_\beta|}

where $\Delta_i$ is the Laplacian w.r.t the i-th electron, $r_i$ is the position of the i-th electron, $R_\alpha$ the position of the $\alpha$-th atom and $Z_\alpha$ its atomic number.
QMC simulations express this integral as:

.. math::

    E = \int \rho(R)E_L(R)dR \geq E_0

with:
.. math::
    \rho(R) = \frac{|\Psi_\theta(R)|^2}{\int |\Psi_\theta(R)|^2} dR }

reprensent the denisty associated with the wave function, and:

.. math::

    E_L(R) = \frac{H\Psi_theta(R)}{\Psi_theta(R)}

are the so called local energies of the system. QMC simulation then approximated the total energy as:

.. math::
    E \approx \frac{1}{M}\sum_i^M \frac{H\Psi_theta(R_i)}{\Psi_theta(R_i)}

where $R_i$ are samples of the density $\rho$ for example obtained via Metropolis Hasting sampling. 
QMC simulations rely then on the optimization of the variational parameters of the wave function, $\theta$, to minimize the value
of the total energy of the system.

.. image:: ../../pics/qmc.png