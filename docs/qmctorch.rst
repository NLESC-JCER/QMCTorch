QMCTorch
===========================================

`QMCTorch` allows to epxress the wave function ususally used by QMC practitioner as neural network. The most generic architecture of the
neural network used by the code is:

.. image:: ../../paper/qmctorch2.png


Through a series of transformation detailled below the network computed the wave function:

.. math::
    \Psi(R) = J(R) \sum_n c_n D_n^{\uparrow} D_n^{\downarrow}

The term `J(R)` is the so called Jastrow factor that captures the electronic correlation. The Jastrow factor is given by :

.. math::

    J(R) = \exp\left(  \sum_{i<j} \text{Kernel}(r_{ij}) \right)

where the sum runs over all the electron pairs and where the kernel function defines the action of the Jastrow factor. A common expression for the
kernel (and the default option for QMCTorch) is the Pade-Jastrow form given by:

.. math::

    \text{Kernel}(r_{ij}) = \frac{\omega_0 r_{ij}}{1+\omega r_{ij}}

where :math: `\omega_0` is a fixed coefficient equals to 0.25(0.5) for antiparallel(parallel) electron spins and :math: `\omega` a variational parameter.

The determinantal parts in the expression of :math: `\Psi` are given by the spin-up and spin-down slater determinants e.g. :

.. math::

    D_n^{\uparrow} = \frac{1}{\sqrt{N}} \begin{vmatrix} & & \\ & \phi_j(r_i) & \\ & & \end{vmatrix}
