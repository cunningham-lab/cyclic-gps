#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "pytorch-lightning",
    "torch",
    "matplotlib",
    "scipy",
    "black",
    "pytest",
    "torchtyping",
    "typeguard",
]


setup(
    name="cyclic-gps",
    packages=find_packages(),
    version=version,
    description="Cyclic reduction for leggps",
    author="Dan Biderman",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="danbider@gmail.com",
    url="https://github.com/cunningham-lab/cyclic-gps",
    keywords=["machine learning", "gaussian processes"],
)
