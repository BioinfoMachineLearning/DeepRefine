#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='DeepRefine',
    version='1.0.0',
    description='A geometric deep learning method for refining protein complex structures.',
    author='Alex Morehead',
    author_email='acmwhb@umsystem.edu',
    license='GNU Public License, Version 3.0',
    url='https://github.com/BioinfoMachineLearning/DeepRefine',
    install_requires=[
        'setuptools==59.5.0',
        'atom3-py3==0.2.1',
        'easy-parallel-py3==0.1.6.4',
        'biopandas==0.2.9',
        'pdb_tools==2.4.5',
        'plotly==5.6.0',
        'matplotlib==3.5.1',
        'dill==0.3.4',
        'tqdm==4.63.1',
        'torchmetrics==0.7.3',
        'wandb==0.12.16',
        'pytorch-lightning==1.5.10',
        'fairscale==0.4.6',
        'deepspeed==0.5.8',
        'absl-py==0.13.0',
        'docker==5.0.2',
        'e3nn==0.4.4',
        'pynvml==11.0.0',
        'axial-positional-embedding==0.2.1',
        'einops==0.4.1',
        'linformer==0.2.1',
        'local_attention==1.4.3',
        'product-key-memory==0.1.10'
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
