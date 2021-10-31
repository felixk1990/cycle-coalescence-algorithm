# @Author:  Felix Kramer
# @Date:   2021-01-14T22:42:23+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-24T22:39:56+02:00
# @License: MIT


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "cycle_analysis", # Replace with your own username
    version = "0.0.4",
    author = "felixk1990",
    author_email = "felixuwekramer@protonmail.com",
    description = "cycle_analysis module, performing minimal cycle basis calculation and the cycle coalescecne algorithm.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/felixk1990/cycle-coalescence-algorithm",
    packages=setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)

from setuptools import setup, find_packages
