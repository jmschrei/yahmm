from distutils.core import setup

setup(
    name='yahmm',
    version='0.1.1',
    author='Adam Novak, Jacob Schreiber',
    author_email='anovak1@ucsc.edu, jmschreiber91@gmail.com',
    packages=['yahmm'],
    scripts=['bin/example.py'],
    url='http://pypi.python.org/pypi/yahmm/',
    license='LICENSE.txt',
    description='HMM package which you build node by node and edge by edge.',
    long_description=open('README.txt').read(),
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "scipy >= 0.13.3",
        "networkx >= 1.8.1",
        "matplotlib >= 1.3.1"
    ],
)