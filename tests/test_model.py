from __future__ import  (division, print_function)

from yahmm.yahmm import *
from nose.tools import with_setup

def setup_model():

	random.seed(0)

	s1 = State(UniformDistribution(0.0, 1.0), name="S1")
	s2 = State(UniformDistribution(0.5, 1.5), name="S2")
	s3 = State(UniformDistribution(-1.0, 1.0), name="S3")

	# Make a simple 2-state model
	global model_a
	model_a = Model(name="A")
	model_a.add_state(s1)
	model_a.add_state(s2)
	model_a.add_transition(s1, s1, 0.70)
	model_a.add_transition(s1, s2, 0.25)
	model_a.add_transition(s1, model_a.end, 0.05)
	model_a.add_transition(s2, s2, 0.70)
	model_a.add_transition(s2, s1, 0.25)
	model_a.add_transition(s2, model_a.end, 0.05)
	model_a.add_transition(model_a.start, s1, 0.5)
	model_a.add_transition(model_a.start, s2, 0.5)

	# Make another model with that model as a component
	global model_b
	model_b = Model(name="B")
	model_b.add_state(s3)
	model_b.add_transition(model_b.start, s3, 1.0)
	model_b.add_model(model_a)
	model_b.add_transition(s3, model_a.start, 1.0)
	model_b.add_transition(model_a.end, model_b.end, 1.0)
	model_b.bake()

def teardown_model():
	pass # just pass for now

@with_setup(setup_model, teardown_model)
def test_sampling():

	sample = model_b.sample()

	assert all([val in sample for val in [0.515908805880605, 1.0112747213686086,
		1.2837985890347725, 0.9765969541523558, 1.4081128851953353,
		0.7818378443997038, 0.6183689966753316]]) , "Sampling check"

@with_setup(setup_model, teardown_model)
def test_viterbi():
	# This sequence should work
	viterbi_output = model_b.viterbi([-0.5, 0.2, 0.2])

	assert viterbi_output[0] == -4.738701578612614, "Viterbi check"

	# This one should not
	viterbi_output = model_b.viterbi([-0.5, 0.2, 0.2 -0.5])
	assert str(viterbi_output) == "(-inf, None)", "Impossible sequence Viterbi check"

@with_setup(setup_model, teardown_model)
def test_training():
	training_improvement = model_b.train([[-0.5, 0.2, 0.2], [-0.5, 0.2, 1.2, 0.8]],
		                                 transition_pseudocount=1)

	assert str(training_improvement) == str(4.13320746579), "Model training improvement"
