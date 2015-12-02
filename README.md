yahmm
=====

[![Build Status](https://travis-ci.org/jmschrei/yahmm.svg?branch=master)](https://travis-ci.org/jmschrei/yahmm)

Yet Another Hidden Markov Model library

<b>NOTE: While yahmm is still fully functional, active development has moved over to  [pomegranate](https://github.com/jmschrei/pomegranate). Please switch over at your convenience.</b>

This module implements Hidden Markov Models (HMMs) with a compositional, graph-
based interface. Models can be constructed node by node and edge by edge, built
up from smaller models, loaded from files, baked (into a form that can be used
to calculate probabilities efficiently), trained on data, and saved.

Implements the forwards, backwards, forward-backward, and Viterbi algorithms, 
and training by both Baum-Welch and Viterbi algorithms.

Silent states are accounted for, but loops containing all silent states are
prohibited. Tied states are also implemented, and handled appropriately in
the training of models.

## Installation

Since yahmm is on PyPi, installation is as easy as running

```
pip install yahmm
```

## Contributing

If you would like to contribute a feature then fork the master branch (fork the release if you are fixing a bug). Be sure to run the tests before changing any code. You'll need to have [nosetests](https://github.com/nose-devs/nose) installed. The following command will run all the tests:
```
nosetests -w tests/
```
Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions. 

## Documentation

See the [wiki](https://github.com/jmschrei/yahmm/wiki) for documentation of yahmm's functions and design. For real-world usage check out the [examples](http://nbviewer.ipython.org/github/jmschrei/yahmm/tree/master/examples/). 

### Tutorial

For our examples here we're going to make the random number generator 
deterministic:

```
>>> random.seed(0)
```

To use this module, first create a Model, which is the main HMM class:

```
>>> model = Model(name="ExampleModel")
```

You then need to populate the Model with State objects. States are constructed 
from emission distributions; right now a few continuous distributions over 
floats are available, but new Distribution classes are simple to write. For our 
example, we will use the UniformDistribution:

```
>>> distribution = UniformDistribution(0.0, 1.0)
```

And then construct a state that emits from the distribution:

```
>>> state = State(distribution, name="uniform")
```

And another state, emitting from a normal distribution with mean 0 and standard 
deviation 2:

```
>>> state2 = State(NormalDistribution(0, 2), name="normal")
```

If None is used as the distribution when creating a state, that state is a 
"silent state". Silent states don't emit anything, but are useful for wiring 
together complex HMMs. By default, a model has two special silent states: a 
start state Model.start, and an end state Model.end.

Topologies which include cycles of only silent states are prohibited; most HMM 
algorithms cannot process them.

```
>>> silent = State(None, name="silent")
```

We then add states to the HMM with the Model.add_state method:

```
>>> model.add_state(state)
>>> model.add_state(state2)
```

You can then add transitions between states, with associated probabilities.
Out-edge probabilities are normalized to 1. for every state when the model is
baked, not before. 

```
>>> model.add_transition(state, state, 0.4)
>>> model.add_transition(state, state2, 0.4)
>>> model.add_transition(state2, state2, 0.4)
>>> model.add_transition(state2, state, 0.4)
```

Don't forget transitions in from the start state and out to the end state:

```
>>> model.add_transition(model.start, state, 0.5)
>>> model.add_transition(model.start, state2, 0.5)
>>> model.add_transition(state, model.end, 0.2)
>>> model.add_transition(state2, model.end, 0.2)
```

If you want to look at your model, try Model.draw(). Note that this 
unfortunately cannot plot self loops. If you want to do a better job of drawing 
the model, the underlying HMM graph is accessible as the graph attribute of the 
model object.

If you want to compose two Models together, use the Model.add_model() method. 
Note that you should not try to use the added model separately once you do this.
You can also make use of the Model.concatenate_model() method, which will assume
you simply want to connect model_a.end to model_b.start with a 1. probability
edge. 

Once we've finished building our model, we have to bake it. Internally, baking 
the model generates the transition log probability matrix, and imposes a 
numerical ordering on the states. If you add more states to the model, you will 
have to bake it again. Baking is also where edge normalization occurs to ensure
that the out-edges for all nodes (except Model.end) sum to 1. Lastly, a
simplification of the graph occurs here, merging any silent states which are
connected simply by a 1.0 probability edge, as they cannot add value to the
graph. You may toggle 'verbose=True' in the bake method to get a log of
when either change occurs to your graph.

```
>>> model.bake()
```

Now that our model is complete, we can generate an example sequence from it:

```
>>> sequence = model.sample()
>>> sequence
[0.7579544029403025, 0.25891675029296335, 0.4049341374504143, \
0.30331272607892745, 0.5833820394550312]
```
And another:

```
>>> model.sample()
[0.28183784439970383, 0.6183689966753316, -2.411068768608379]
```

And another:

```
>>> model.sample()
[0.47214271545271336, -0.5804485412450214]
```

We can calculate the log probability of the sequence given the model (the log 
likelihood), summing over all possible paths, using both the forward and 
backward algorithms. Log probability is reported in nats (i.e. it is natural 
log probability). Both algorithms return the full table of size
len( observations ) x len( states ). For the forward algorithm, the entry
at position i, j represents the log probability of beginning at the start 
of the sequence, and summing over all paths to align observation i to hidden
state j. This state can be recovered by pulling it from model.states[j].

```
>>> model.forward(sequence)
[[       -inf        -inf        -inf  0.        ]
 [-2.37704475 -0.69314718 -2.1322948         -inf]
 [-3.05961307 -1.43914762 -2.86809348        -inf]
 [-3.80752847 -2.1749463  -3.60588302        -inf]
 [-4.53632138 -2.91273584 -4.34219628        -inf]
 [-5.30367664 -3.6490491  -5.08355666        -inf]]
```

In order to get the log probability of the full sequence given the model,
you can write the following:

```
>>> model.forward(sequence)[ len(sequence), model.end_index ]
-5.0835566645
```

Or, use a wrapper to get that value by default:

```
>>> model.log_probability(sequence)
-5.0835566645
```

The same paradigm is used for the backward algorithm. Indices i, j represent
the probability of having aligned observation i to state j and continued
aligning the remainder of the sequence till the end.

```
>>> model.backward(sequence)
[[-5.30670022 -5.30670022        -inf -5.08355666]
 [-4.56069977 -4.56069977        -inf -4.33755622]
 [-3.8249011  -3.8249011         -inf -3.60175755]
 [-3.08711156 -3.08711156        -inf -2.863968  ]
 [-2.3507983  -2.3507983         -inf -2.12765475]
 [-1.60943791 -1.60943791  0.                -inf]]

>>> model.backward(sequence)[ 0, model.start_index ]
-5.0835566645
```

The forward-backward algorithm is also implemented in a similar manner. It
will return a tuple of the estimated transition probabilities given with that
sequence and the table of log probabilities of the sum of all paths of the
alignment of observation i with state j. Indices i, j represent having started
at the beginning of the sequence, aligned observation i to state j, and then 
continued on to align the remainder of the sequence to the model.

```
>>> model.forward_backward(sequence)
(array([[-2.03205947, -0.39913252, -1.61932212,        -inf],
       [-2.03481952, -0.40209763, -1.60753724,        -inf],
       [       -inf,        -inf,        -inf,        -inf],
       [-1.85418786, -0.17029029,        -inf,        -inf]]), 
array([[-1.85418786, -0.17029029,  0.        ,  0.        ],
       [-1.80095751, -0.18049206,  0.        ,  0.        ],
       [-1.81108336, -0.17850119,  0.        ,  0.        ],
       [-1.80356301, -0.17997747,  0.        ,  0.        ],
       [-1.82955788, -0.17493035,  0.        ,  0.        ]]))
```

We can also find the most likely path, and the probability thereof, using the 
Viterbi algorithm. This returns a tuple of the likelihood under the ML path and 
the ML path itself. The ML path is in turn a list of tuples of State objects and
the number of items in the sequence that had been generated by that point in the
path (to account for the presence of silent states).

```
>>> model.viterbi(sequence)
(-5.9677480204906654, \
[(0, State(ExampleModel-start, None)), \
(1, State(uniform, UniformDistribution(0.0, 1.0))), \
(2, State(uniform, UniformDistribution(0.0, 1.0))), \
(3, State(uniform, UniformDistribution(0.0, 1.0))), \
(4, State(uniform, UniformDistribution(0.0, 1.0))), \
(5, State(uniform, UniformDistribution(0.0, 1.0))), \
(5, State(ExampleModel-end, None))])
```

Given a list of sequences, we can train our HMM by calling Model.train(). This 
returns the final log score: the log of the sum of the probabilities of all 
training sequences. It also prints the improvement in log score on each training
iteration, and stops if the improvement gets too small or actually goes 
negative. 
```
>>> sequences = [sequence]
>>> model.log_probability(sequence)
-5.0835566644993735

>>> log_score = model.train(sequences)
Training improvement: 5.81315226327
Training improvement: 0.156159401683
Training improvement: 0.0806734819188
Training improvement: 0.0506679952827
Training improvement: 0.142593661095
Training improvement: 0.305806209012
Training improvement: 0.301331333752
Training improvement: 0.380117757466
Training improvement: 0.773814416569
Training improvement: 1.58660096053
Training improvement: 0.439182120777
Training improvement: 0.0067603436265
Training improvement: 5.5971526649e-06
Training improvement: 3.75166564481e-12

>>> model.log_probability(sequence)
-4.9533088776424528
```

In addition to the Baum-Welch algorithm, viterbi training is also included. 
This training is quicker, but less exact than the Baum-Welch algorithm. It 
makes the probability of a transition equal to the frequency of seeing that
transition in the viterbi path of all the training sequences, and emissions
to be the distribution retrained on all obervations tagged with that state
in the viterbi path.

Model.train is a wrapper for both the Viterbi and Baum-Welch algorithms,
which can be specified with "algorithm='Baum-Welch'" or "algorithm='Viterbi'".
The Baum-Welch algorithm can also take min_iterations to do at least that any
iterations of Baum-Welch training, and stop_threshold to indicate the log score
improvement at which to stop at-- currently set at 1e-9. Viterbi training takes
no arguments. 

Lastly, tied states are supported in both training algorithms. This is useful
if many states are supposed to represent the same underlying distribution, which
should be kept the same even upon being retrained. When not tied, these states
may diverge slightly from each other. Tying them both keeps them all the same,
and increases the amount of training data each distribution gets, to hopefully
get a better result.

In order to use a tied state, simply pass the same distribution object into
multiple states. See the following example.

```
# NOT TIED STATES
>>> a = State( NormalDistribution( 5, 2 ), name="A" )
>>> b = State( UniformDistribution( 2, 7 ), name="B" )
>>> c = State( NormalDistribution( 5, 2 ), name="C" )

# A AND C TIED STATES
>>> d = NormalDistribution( 5, 2 )
>>>
>>> a = State( d, name="A" )
>>> b = State( UniformDistribution( 2, 7 ), name="B" )
>>> c = State( d, name="C" )
```

Once you're done working with your model, you can write it out to a stream with 
Model.write(), to be read back in later with Model.read().

```
>>> model.write(sys.stdout)
ExampleModel 4
302687936 ExampleModel-end 1.0 None
302688008 ExampleModel-start 1.0 None
302688080 normal 1.0 NormalDistribution(0.281114738186, 0.022197987893)
302688152 uniform 1.0 UniformDistribution(0.258916750293, 0.75795440294)
uniform uniform 6.02182522366e-25 0.4 302688152 302688152
uniform ExampleModel-end 0.333333333333 0.2 302688152 302687936
uniform normal 0.666666666667 0.4 302688152 302688080
normal uniform 1.0 0.4 302688080 302688152
normal ExampleModel-end 9.71474187173e-184 0.2 302688080 302687936
normal normal 2.59561866186e-45 0.4 302688080 302688080
ExampleModel-start uniform 1.0 0.5 302688008 302688152
ExampleModel-start normal 0.0 0.5 302688008 302688080
```

This file contains states, and then transitions. The first line is the name of the model and the number of states present. Then, each line contains a single state containing a unique ID, the name, the state weight, and the distribution that the state contains. For the start and end state, this value is None, as they are silent states. Then, the remaining lines contain transitions in the model, formatted by from\_state\_name, to\_state\_name, probability, pseudocount, from\_state\_id, and to\_state\_id. The IDs are unique tags generated from the memory address of the state, and are needed in case the user names two states with the same name. As an example, the first transition is from the state named uniform to the state named uniform with a very low probability, and the IDs are the same meaning that it is a self loop. 

Lets explore the bake method a little more. In addition to finalizing the
internal structure of the model, it will normalize out-edge weights, and also
merge silent states with a probability 1. edge between them to simplify the
model. Lets see this in action.
```
	model_a = Model( "model_a" )
	s1 = State( NormalDistribution( 25., 1. ), name="S1" )
	s2 = State( NormalDistribution( 13., 1. ), name="S2" )

	model_a.add_state( s1 )
	model_a.add_state( s2 )
	model_a.add_transition( model.start, s1, 0.95 )
	model_a.add_transition( s1, s1, 0.40 )
	model_a.add_transition( s1, s2, 0.50 )
	model_a.add_transition( s2, s1, 0.50 )
	model_a.add_transition( s2, s2, 0.40 )
	model_a.add_transition( s1, model.end, 0.1 )
	model_a.add_transition( s2, model.end, 0.1 )

	model_b = Model( "model_b" )
	s3 = State( NormalDistribution( 34., 1. ), name="S3" )
	s4 = State( NormalDistribution( 45., 1. ), name="S4" )

	model_b.add_state( s3 )
	model_b.add_state( s4 )
	model_b.add_transition( model.start, s3, 1.0 )
	model_b.add_transition( s3, s3, 0.50 )
	model_b.add_transition( s3, s4, 0.30 )
	model_b.add_transition( s4, s4, 0.20 )
	model_b.add_transition( s4, s3, 0.30 )
	model_b.add_transition( s4, model.end, 1.0 )
```
If at this point we baked model_a and ran it, we'd get the following:
```
>>> sequence = [ 24.57, 23.10, 11.56, 14.3, 36.4, 33.2, 44.2, 46.7 ]
>>> model_a.bake( verbose=True )
model_a : model_a-start summed to 0.95, normalized to 1.0
>>>
>>> print model_a.forward( sequence )
[[         -inf          -inf          -inf    0.        ]
 [         -inf   -1.01138853   -3.31397363          -inf]
 [ -53.62847425   -4.6516178    -6.95420289          -inf]
 [  -7.30050351  -96.80364706   -9.60308861          -inf]
 [  -9.98073278  -66.15758923  -12.28331787          -inf]
 [-285.59596204  -76.57281849  -78.87540358          -inf]
 [-282.2049042  -112.02804776 -114.33063285          -inf]
 [-600.36013347 -298.18327702 -300.48586211          -inf]
 [-867.64036273 -535.46350629 -537.76609138          -inf]]
>>> 
>>> print model_a.log_probability( sequence )
-537.766091379
```
By setting verbose=True, we get a log that the out-edges from model.start have
been normalized to 1.0. This forward log probability matrix is the same as if
we had originally set the transition to 1.0

If instead of the above, we concatenated the models and ran the code, we'd
get the following:
```
>>> sequence = [ 24.57, 23.10, 11.56, 14.3, 36.4, 33.2, 44.2, 46.7 ]
>>> model_a.concatenate_model( model_b )
>>> model_a.bake( verbose=True )
model_a : model_a-end (silent) - model_b-start (silent) merged
model_a : model_a-start summed to 0.95, normalized to 1.0
model_a : S3 summed to 0.8, normalized to 1.0
>>> 
>>> print model_a.forward( sequence )
[[         -inf          -inf          -inf          -inf          -inf		
 -inf            0.]
 [         -inf   -1.01138853          -inf          -inf   -3.31397363	
 	 -inf          -inf]
 [ -63.63791216   -4.6516178           -inf  -53.62847425   -6.95420289	
 	 -inf          -inf]
 [-259.64994142  -96.80364706 -624.65447995   -7.30050351   -9.60308861 
 -624.65447995          -inf]
 [-204.56702714  -66.15758923 -732.79470921   -9.98073278  -12.28331787	
  	 -inf          -inf]
 [ -16.0822564   -76.57281849 -243.44679492 -285.59596204  -78.87540358	
 -243.44679492          -inf]
 [ -17.79119857 -112.02804776  -87.60202419 -282.2049042  -114.33063285 
  -87.60202419          -inf]
 [ -71.20014073 -298.18327702  -20.01096635 -600.36013347 -300.48586211 
  -20.01096635          -inf]
 [-102.77887769 -535.46350629  -23.9843428  -867.64036273 -537.76609138 
   -23.9843428          -inf]]

>>>
>>> print model_a.log_probability( sequence )
-23.9843427976
```
We see both bake processing operations in effect. Both model_a.start and S3 did
not have properly summed out-edges, and needed to have them normalized. But now
there was a useless edge between model_a.end and model_b.start due to the
concatenate method. This allowed those two states to be merged, speeding up
later algorithms. We can also see that the addition of model_b made the sequence
significantly more likely given the model, as a sanity check that
concatenate_model really did work.

As said above, this module provides a few distributions over floats by default:

UniformDistribution( start, end )

NormalDistribution( mean, std )

ExponentialDistribution( rate )

GammaDistribution( shape, rate ) 
(Note that this differs from the parameterization used in the random module, 
even though the parameters have the same names.

InverseGammaDistribution( shape, rate )

GaussianKernelDensity( points, bandwidth, weights=None )

UniformKernelDensity( points, bandwidth, weights=None )

TriangleKernelDensity( points, bandwidth, weights=None )

MixtureDistribution( distributions, weights=None )

The module also provides two other distributions:

DiscreteDistribution( characters )
( Allows you to pass in a dictionary of key: probability pairs )

LambdaDistribution( lambda_funct )
( Allows you to pass in an arbitrary function that returns a log probability for
a given symbol )

To add a new Distribution, with full serialization and deserialization support, 
you have to make a new class that inherits from Distribution. That class must 
have:

	* A class-level name attribute that is unique amoung all distributions, and 
	  is used for serialization.
	* An __init__ method that stores all constructor arguments into 
	  self.parameters as a list.
	* A log_probability method that returns the log of the probability of the 
	  given value under the distribution, and which reads its parameters from 
	  the self.parameters list. This module's log() and exp() functions can be 
	  used instead of the default Python ones; they handle numpy arrays and 
	  "properly" consider the log of 0 to be negative infinity.
	* A from_sample method, which takes a Numpy array of samples and an optional
	  Numpy array of weights, and re-estimates self.parameters to maximize the 
	  likelihood of the samples weighted by the weights. Note that weighted 
	  maximum likelihood estimation can be somewhat complicated for some 
	  distributions (see, for example, the GammaDistribution here).
	* A sample method, which returns a randomly sampled value from the 
	  distribution.


The easiest way to define a new distribution is to just copy-paste the 
UniformDistribution from the module and replace all its method bodies.
Any distribution you define can be easily plugged in with other
distributions, assuming that it has the correct methods. However, if
you write the model and give it to someone else, they might not have
the custom distribution.

Here is an example discrete distribution over {True, False}:
```
>>> class BernoulliDistribution(Distribution):
...     name = "BernoulliDistribution"
...     def __init__(self, p):
...         self.parameters = [p]
...     def log_probability(self, sample):
...         if sample:
...             return log(self.parameters[0])
...         else:
...             return log(1 - self.parameters[0])
...     def from_sample(self, items, weights=None):
...         if weights is None:
...             weights = numpy.ones_like(items, dtype=float)
...         self.parameters = [float(numpy.dot(items, weights)) / len(items)]
...     def sample(self):
...         return random.random() < self.parameters[0]
>>> bernoulli = BernoulliDistribution(0.5)
>>> exp(bernoulli.log_probability(True))
0.5
>>> sample = [bernoulli.sample() for i in xrange(10)]
>>> sample
[False, True, False, True, False, False, True, False, True, False]
>>> bernoulli.from_sample(sample)
>>> bernoulli.write(sys.stdout)
BernoulliDistribution(0.4)
```
```
	# Test HMMS
	
	model_a = Model(name="A")
	model_b = Model(name="B")
	
	s1 = State(UniformDistribution(0.0, 1.0), name="S1")
	s2 = State(UniformDistribution(0.5, 1.5), name="S2")
	s3 = State(UniformDistribution(-1.0, 1.0), name="S3")
	
	# Make a simple 2-state model
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
	model_b.add_state(s3)
	model_b.add_transition(model_b.start, s3, 1.0)
	model_b.add_model(model_a)
	model_b.add_transition(s3, model_a.start, 1.0)
	model_b.add_transition(model_a.end, model_b.end, 1.0) 
	
	model_b.bake()
	
	print "HMM:"
	print model_b
	
	print "HMM serialization:"
	model_b.write(sys.stdout)
	
	print "A sample from the HMM:"
	print model_b.sample()
	
	print "Forward algorithm:"
	print model_b.forward([]) # Impossible
	print model_b.forward([-0.5, 0.2, 0.2]) # Possible
	print model_b.forward([-0.5, 0.2, 0.2 -0.5]) # Impossible
	print model_b.forward([-0.5, 0.2, 1.2, 0.8]) # Possible
	
	print "Backward algorithm:"
	print model_b.backward([]) # Impossible
	print model_b.backward([-0.5, 0.2, 0.2]) # Possible
	print model_b.backward([-0.5, 0.2, 0.2 -0.5]) # Impossible
	print model_b.backward([-0.5, 0.2, 1.2, 0.8]) # Possible
	
	print "Viterbi:"
	print model_b.viterbi([]) # Impossible
	print model_b.viterbi([-0.5, 0.2, 0.2]) # Possible
	print model_b.viterbi([-0.5, 0.2, 0.2 -0.5]) # Impossible
	print model_b.viterbi([-0.5, 0.2, 1.2, 0.8]) # Possible
	
	# Train on some of the possible data
	print "Training..."
	model_b.train([[-0.5, 0.2, 0.2], [-0.5, 0.2, 1.2, 0.8]], 
		transition_pseudocount=1)
	print "HMM after training:"
	print model_b
	
	print "HMM serialization:"
	model_b.write(sys.stdout)
	
	print "Probabilities after training:"
	print model_b.forward([]) # Impossible
	print model_b.forward([-0.5, 0.2, 0.2]) # Possible
	print model_b.forward([-0.5, 0.2, 0.2 -0.5]) # Impossible
	print model_b.forward([-0.5, 0.2, 1.2, 0.8]) # Possible
```
