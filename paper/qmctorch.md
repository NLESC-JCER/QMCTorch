---
title: 'QMCTorch: Real-Space Quantum Monte Carlo Simulations of Molecular Systems using PyTorch'
tags:
  - Python
  - Deep Learning
  - Quantum Chemistry
  - Monte Carlo
  - Molecular Systems
authors:
  - name: Nicolas Renaud
    orcid: 0000-0001-9589-2694
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Netherlands eScience Center, Science Park 402, 1098 XH Amsterdam, The Netherlands
   index: 1
date: 11 April 2023
bibliography: qmctorch.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Quantum Monte-Carlo (QMC) simulations allow to compute the electronic structure of quantum systems
with high accuracy at a relatively low computational cost. QMC relies on the definition of a wave
function ansatz that is optimized to minimize the total eneregy of the quantum systems. `QMCTorch`
allows to recast this optimization as a deep-learning problem where the wave function ansatz is expressed
as a physically-motivated neural network. The use of PyTorch as a backend to perform the optimization allows 
leveraging automatic differentiation to compute the gradient of the total energy wrt the variational parameters
as well as GPU offloading to accelerate the calculation. `QMCTorch` is interfaced with popular quantum chemistry packages
to facilitate its utilisation and help its adoption by quantum chemists and material scientists.  


# Statement of need

`QMCTorch` is a Python package using PyTorch [paszke2019pytorch] as a backend to perform Quantum Monte-Carlo (QMC) simulations of molecular systems. Many software such as `QMCPack`[@qmcpack], `QMC=Chem` [@qmcchem], `CHAMP` [@champ] provide high-quality implementation of advanced QMC methodologies in low-level languages (C++/Fortran).  Python implementations of QMC such as `PAUXY` [@pauxy] and `PyQMC` [@pyqmc] have also been proposed to facilitate the use and development of QMC techniques. Recently large efforts have been made to leverage recent development of deep learning techniques for QMC simulations. Hence neural-network based wave-function ansatz has been proposed [@paulinet,@ferminet]. These recent advances lead to very interesting results but lack in explainability of the resulting wave function ansatz. `QMCTorch` allows to perform QMC simulations using physically motivated neural netwrok architecture that closely follows the wnave function ansatz used by QMC practitioners. As such, it still allows to leverage automatic differentiation for the calculation of the gradients of the total energy wrt the variational parameters and the GPU capabilites offered by PyTorch without loosing the physical intuition behind the wave function ansatz. 


# Architecture
![General architectureof the neural network used by `QMCTorch` to encodie the wave function ansatz.\label{fig:arch}](qmctorch2.png)

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References