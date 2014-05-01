from setuptools import setup
from distutils.extension import Extension
import numpy as np

"""
Method for importing cython modules described here:
http://stackoverflow.com/questions/4505747/how-should-i-structure-a-
    python-package-that-contains-cython-code
"""
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    name='yahmm',
    version='0.1.10',
    author='Adam Novak, Jacob Schreiber',
    author_email='anovak1@ucsc.edu, jmschreiber91@gmail.com',
    packages=['yahmm'],
    scripts=['bin/example.py'],
    url='http://pypi.python.org/pypi/yahmm/',
    license='LICENSE.txt',
    description='HMM package which you build node by node and edge by edge.',
    long_description=open('README.txt').read(),
    cmdclass={'build_ext':build_ext},
    ext_modules=cythonize([ Extension( "yahmm", ["yahmm/yahmm.pyd"], include_dirs=[np.get_include()] )]),
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "scipy >= 0.13.3",
        "networkx >= 1.8.1",
        "matplotlib >= 1.3.1"
    ],
)
