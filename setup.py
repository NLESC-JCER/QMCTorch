#!/usr/bin/env python

import os

from setuptools import (find_packages, setup)

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit QMCTorch/__version__.py
version = {}
with open(os.path.join(here, 'qmctorch', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='qmctorch',
    version=version['__version__'],
    description="Pytorch Implementation of Quantum Monte Carlo",
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=["Nicolas Renaud", "Felipe Zapata"],
    author_email='n.renaud@esciencecenter.nl',
    url='https://github.com/NLESC-JCER/QMCTorch',
    packages=find_packages(),
    package_dir={'qmctorch': 'qmctorch'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='qmctorch',
    scripts=['bin/qmctorch'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    test_suite='tests',
    install_requires=['matplotlib', 'numpy', 'argparse',
                      'scipy', 'tqdm', 'torch', 'plams',
                      'pyscf', 'mendeleev', 'twiggy', 'mpi4py'],

    extras_require={
        'hpc': ['horovod==0.27.0'],
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme', 'nbsphinx'],
        'test': ['pytest', 'pytest-runner',
                 'coverage', 'coveralls', 'pycodestyle'],
    }
)
