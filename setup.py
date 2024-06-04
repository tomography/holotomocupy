from setuptools import setup, find_packages
from setuptools.command.install import install
import os


setup(
    name='holotomo',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    author_email='nikitinvv@anl.gov',
    url='https://github.com/nikitinvv/holotomo',
    packages=find_packages(),
    include_package_data = True,
    description='Tools for iterative holotomography reconstruction',
    zip_safe=False,
)