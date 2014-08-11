from __future__ import  (division, print_function)

from yahmm.yahmm import *
from nose.tools import with_setup
import random
import numpy as np

def setup():
	'''
	No setup or teardown needs to be done in this case.
	'''

	pass

def teardown():
	'''
	No setup or teardown needs to be done in this case.
	'''

	pass

@with_setup( setup, teardown )
def test_normal():
	'''
	Test that the normal distribution implementation is correct.
	'''

	d = NormalDistribution( 5, 2 )
	e = NormalDistribution( 5., 2. )

	assert d.log_probability( 5 ) == -1.6120857137642188
	assert d.log_probability( 5 ) == e.log_probability( 5 )
	assert d.log_probability( 5 ) == d.log_probability( 5. )

	assert d.log_probability( 0 ) == -4.737085713764219
	assert d.log_probability( 0 ) == e.log_probability( 0. )

	d.from_sample( [ 5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4 ] )

	assert d.parameters == [ 4.916666666666667, 0.75920279826202286 ]
	assert d.log_probability( 4 ) != e.log_probability( 4 )
	assert d.log_probability( 4 ) == -1.3723678499651766
	assert d.log_probability( 18 ) == -149.13140399454429
	assert d.log_probability( 1e8 ) == -8674697942168743.0

	d = NormalDistribution( 5, 1e-10 )
	assert d.log_probability( 1e100 ) == -4.9999999999999994e+219

	d.from_sample( [ 0, 2, 3, 2, 100 ], weights=[ 0, 5, 2, 3, 200 ] )
	assert d.parameters == [ 95.342857142857142, 20.827558927640887 ]
	assert d.log_probability( 50 ) == -6.325011936564346

@with_setup( setup, teardown )
def test_uniform():
	'''
	Test that the uniform distribution implementation is correct.
	'''

	d = UniformDistribution( 0, 10 )

	assert d.log_probability( 2.34 ) == -2.3025850929940455
	assert d.log_probability( 2 ) == d.log_probability( 8 )
	assert d.log_probability( 10 ) == d.log_probability( 3.4 )
	assert d.log_probability( 1.7 ) == d.log_probability( 9.7 )
	assert d.log_probability( 10.0001 ) == float( "-inf" )
	assert d.log_probability( -0.0001 ) == float( "-inf" )

	for i in xrange( 10 ):
		data = np.random.randn( 100 ) * 100
		d.from_sample( data )
		assert d.parameters == [ data.min(), data.max() ]


	for i in xrange( 100 ):
		sample = d.sample()
		assert data.min() <= sample <= data.max()

@with_setup( setup, teardown )
def test_discrete():
	'''
	Test that the discrete distribution implementation is correct.
	'''

	d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

	assert d.log_probability( 'C' ) == -1.3862943611198906
	assert d.log_probability( 'A' ) == d.log_probability( 'C' )
	assert d.log_probability( 'G' ) == d.log_probability( 'T' )
	assert d.log_probability( 'a' ) == float( '-inf' )

	seq = "ACGTACGTTGCATGCACGCGCTCTCGCGC"
	d.from_sample( list( seq ) )

	assert d.log_probability( 'C' ) == -0.9694005571881033
	assert d.log_probability( 'A' ) == -1.9810014688665833
	assert d.log_probability( 'T' ) == -1.575536360758419

	seq = "ACGTGTG"
	d.from_sample( list( seq ), weights=[0.,1.,2.,3.,4.,5.,6.] )

	assert d.log_probability( 'A' ) == float( '-inf' )
	assert d.log_probability( 'C' ) == -3.044522437723423
	assert d.log_probability( 'G' ) == -0.5596157879354228

@with_setup( setup, teardown )
def test_lognormal():
	'''
	Test that the lognormal distribution implementation is correct.
	'''

	d = LogNormalDistribution( 5, 2 )
	assert d.log_probability( 5 ) == -4.658512534903223

@with_setup( setup, teardown )
def test_gamma():
	'''
	Test that the gamma distribution implementation is correct.
	'''

	d = GammaDistribution( 5, 2 )
	assert d.log_probability( 4 ) == -2.167140483068656

@with_setup( setup, teardown )
def test_exponential():
	'''
	Test that the beta distribution implementation is correct.
	'''

	d = ExponentialDistribution( 3 )
	assert d.log_probability( 8 ) == -22.90138771133189

@with_setup( setup, teardown )
def test_inverse_gamma():
	'''
	Test that the inverse gamma distribution implementation is correct.
	'''

	d = InverseGammaDistribution( 4, 5 )
	assert d.log_probability( 1.06 ) == -0.24579567593905338

@with_setup( setup, teardown )
def test_gaussian_kernel():
	'''
	Test that the Gaussian Kernel Density implementation is correct.
	'''

	d = GaussianKernelDensity( [ 0, 4, 3, 5, 7, 4, 2 ] )
	assert d.log_probability( 3.3 ) == -1.7041719932783466

	d.from_sample( [ 1, 6, 8, 3, 2, 4, 7, 2] )
	assert d.log_probability( 1.2 ) == -2.0237274126309863

	d.from_sample( [ 1, 0, 108 ], weights=[2., 3., 278.] )
	assert d.log_probability( 110 ) == -2.9367643171568742
	assert d.log_probability( 0 ) == -5.126195893144003

@with_setup( setup, teardown )
def test_triangular_kernel():
	'''
	Test that the Triangular Kernel Density implementation is correct.
	'''

	d = TriangleKernelDensity( [ 1, 6, 3, 4, 5, 2 ] )
	assert d.log_probability( 6.5 ) == -2.4849066497880004

@with_setup( setup, teardown )
def test_uniform_kernel():
	'''
	Test that the Uniform Kernel Density implementation is correct.
	'''

	d = UniformKernelDensity( [ 1, 3, 5, 6, 2, 2, 3, 2, 2 ] )

	assert d.log_probability( 2.2 ) == -0.4054651081081643
	assert d.log_probability( 6.2 ) == -2.1972245773362196
	assert d.log_probability( 10 ) == float( '-inf' )

@with_setup( setup, teardown )
def test_mixture():
	'''
	Test that the Mixture Distribution implementation is correct.
	'''

	d = MixtureDistribution( [ NormalDistribution( 5, 1 ), 
							   NormalDistribution( 4, 4 ) ] )

	assert d.log_probability( 6 ) == -1.8018490294558223
	assert d.log_probability( 5 ) == -1.3951145254551627
	assert d.log_probability( 4.5 ) == -1.4893802468206498

	d = MixtureDistribution( [ NormalDistribution( 5, 1 ),
	                           NormalDistribution( 4, 4 ) ],
	                         weights=[1., 7.] )

	assert d.log_probability( 6 ) == -2.2324936044580177
	assert d.log_probability( 5 ) == -2.006552218742595
	assert d.log_probability( 4.5 ) == -2.035633980640667

@with_setup( setup, teardown )
def test_multivariate():
	'''
	Test that the Multivariate Distribution implementation is correct.
	'''

	d = MultivariateDistribution( [ NormalDistribution( 5, 2 ), ExponentialDistribution( 2 ) ] )

	assert d.log_probability( (4,1) ) == -3.0439385332042734
	assert d.log_probability( ( 100, 0.001 ) ) == -1129.0459385332042

	d = MultivariateDistribution( [ NormalDistribution( 5, 2 ), ExponentialDistribution( 2 ) ],
								  weights=[18., 1.] )

	assert d.log_probability( (4,1) ) == -32.574395667195994
	assert d.log_probability( (100, 0.001) ) == -20334.576395667194



