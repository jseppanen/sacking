#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='sacking',
    version='0.0.1.dev0',
    license='Apache-2.0',
    description='Yet another soft actor-critic implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jarno Seppänen',
    author_email='sacking@meit.si',
    url='https://github.com/jseppanen/sacking',
    packages=[
        'sacking',
    ],
    install_requires=[
        'numpy',
        'torch',
        'click',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
