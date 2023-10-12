#! /usr/bin/python3
from setuptools import setup

setup(
    name="AmpliTools",
    version="1.0",
    py_modules=["amplitools"],
    install_requires=["nautypy", "hashable_containers", "sympy", "networkx", "matplotlib"],
)
