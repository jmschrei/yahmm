from distutils.core import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }

if use_cython:
    ext_modules = [
        Extension("yahmm.yahmm", [ "yahmm/yahmm.pyx" ], include_dirs=[np.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [
        Extension("yahmm.yahmm", [ "yahmm/yahmm.c" ], include_dirs=[np.get_include()]),
    ]

script_names = [ 'example.py', 'rainy_sunny_hmm.py', 'tied_state_hmm.py', 
    'infinite_hmm.py']

scripts = [ "examples/{}".format( name ) for name in script_names ]

setup(
    name='yahmm',
    version='0.1.5',
    author='Adam Novak, Jacob Schreiber',
    author_email='anovak1@ucsc.edu, jmschreiber91@gmail.com',
    packages=['yahmm'],
    scripts=scripts,
    url='http://pypi.python.org/pypi/yahmm/',
    license='LICENSE.txt',
    description='HMM package which you build node by node and edge by edge.',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "scipy >= 0.13.3",
        "networkx >= 1.8.1",
        "matplotlib >= 1.3.1"
    ],
)
