# The CLRS Algorithmic Reasoning Benchmark

Learning representations of algorithms is an emerging area of machine learning,
seeking to bridge concepts from neural networks with classical algorithms. The
CLRS Algorithmic Reasoning Benchmark (CLRS) consolidates and extends previous
work toward evaluation algorithmic reasoning by providing a suite of
implementations of classical algorithms. These algorithms have been selected
from the third edition of the standard *Introduction to Algorithms* by Cormen,
Leiserson, Rivest and Stein.

## Getting started

The CLRS Algorithmic Reasoning Benchmark can be installed with pip, either from
PyPI:

```shell
pip install dm-clrs
```

or directly from GitHub (updated more frequently):

```shell
pip install git+git://github.com/deepmind/clrs.git
```

You may prefer to install it in a virtual environment if any requirements
clash with your Python installation:

```shell
python3 -m venv clrs_env
source clrs_env/bin/activate
pip install git+git://github.com/deepmind/clrs.git
```

Once installed you can run our example baseline model:

```shell
python3 -m clrs.examples.run
```

If this is the first run of the example, the dataset will be downloaded and
stored in `--dataset_path` (default '/tmp/CLRS30').
Alternatively, you can also download and extract https://storage.googleapis.com/dm-clrs/CLRS30_v1.0.0.tar.gz

## Algorithms as graphs

CLRS implements the selected algorithms in an idiomatic way, which aligns as
closely as possible to the original CLRS 3ed pseudocode. By controlling the
input data distribution to conform to the preconditions we are able to
automatically generate input/output pairs. We additionally provide trajectories
of "hints" that expose the internal state of each algorithm, to both optionally
simplify the learning challenge and to distinguish between different algorithms
that solve the same overall task (e.g. sorting).

In the most generic sense, algorithms can be seen as manipulating sets of
objects, along with any relations between them (which can themselves be
decomposed into binary relations). Accordingly, we study all of the algorithms
in this benchmark using a graph representation. In the event that objects obey a
more strict ordered structure (e.g. arrays or rooted trees), we impose this
ordering through inclusion of predecessor links.

## How it works

For each algorithm, we provide a canonical set of *train*, *eval* and *test*
trajectories for benchmarking out-of-distribution generalization.

|       | Trajectories    | Problem Size |
|-------|-----------------|--------------|
| Train | 1000            | 16           |
| Eval  | 32 x multiplier | 16           |
| Test  | 32 x multiplier | 64           |


Here, "problem size" refers to e.g. the length of an array or number of nodes in
a graph, depending on the algorithm. "multiplier" is an algorithm-specific
factor that increases the number of available *eval* and *test* trajectories
to compensate for paucity of evaluation signals. "multiplier" is 1 for all
algorithms except:

- Maximum subarray (Kadane), for which "multiplier" is 32.
- Quick select, minimum, binary search, string matchers (both naive and KMP),
and segment intersection, for which "multiplier" is 64.

The trajectories can be used like so:

```python
train_ds, num_samples, spec = clrs.create_dataset(
      folder='/tmp/CLRS30', algorithm='bfs',
      split='train', batch_size=32)

for i, feedback in enumerate(train_ds.as_numpy_iterator()):
  if i == 0:
    model.init(feedback.features, initial_seed)
  loss = model.feedback(rng_key, feedback)
```

Here, `feedback` is a `namedtuple` with the following structure:

```python
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
```

where the content of `Features` can be used for training and `outputs` is
reserved for evaluation. Each field of the tuple is an `ndarray` with a leading
batch dimension. Because `hints` are provided for the full algorithm trajectory,
these contain an additional time dimension padded up to the maximum length
`max(T)` of any trajectory within the dataset. The `lengths` field specifies the
true length `t <= max(T)` for each trajectory, which can be used e.g. for loss
masking.

The `examples` directory contains a full working Graph Neural Network (GNN)
example using JAX and the DeepMind JAX Ecosystem of libraries. It allows
training of multiple algorithms on a single processor, as described in
["A Generalist Neural Algorithmic Learner"](https://arxiv.org/abs/2209.11142).

## What we provide

### Algorithms

Our initial CLRS-30 benchmark includes the following 30 algorithms. We aim to
support more algorithms in the future.

- Sorting
  - Insertion sort
  - Bubble sort
  - Heapsort (Williams, 1964)
  - Quicksort (Hoare, 1962)
- Searching
  - Minimum
  - Binary search
  - Quickselect (Hoare, 1961)
- Divide and conquer
  - Maximum subarray (Kadane's variant) (Bentley, 1984)
- Greedy
  - Activity selection (Gavril, 1972)
  - Task scheduling (Lawler, 1985)
- Dynamic programming
  - Matrix chain multiplication
  - Longest common subsequence
  - Optimal binary search tree (Aho et al., 1974)
- Graphs
  - Depth-first search (Moore, 1959)
  - Breadth-first search (Moore, 1959)
  - Topological sorting (Knuth, 1973)
  - Articulation points
  - Bridges
  - Kosaraju's strongly connected components algorithm (Aho et al., 1974)
  - Kruskal's minimum spanning tree algorithm (Kruskal, 1956)
  - Prim's minimum spanning tree algorithm (Prim, 1957)
  - Bellman-Ford algorithm for single-source shortest paths (Bellman, 1958)
  - Dijkstra's algorithm for single-source shortest paths (Dijkstra et al., 1959)
  - Directed acyclic graph single-source shortest paths
  - Floyd-Warshall algorithm for all-pairs shortest-paths (Floyd, 1962)
- Strings
  - Naïve string matching
  - Knuth-Morris-Pratt (KMP) string matcher (Knuth et al., 1977)
- Geometry
  - Segment intersection
  - Graham scan convex hull algorithm (Graham, 1972)
  - Jarvis' march convex hull algorithm (Jarvis, 1973)

### Baselines

Models consist of a *processor* and a number of *encoders* and *decoders*.
We provide JAX implementations of the following GNN baseline processors:

- Deep Sets (Zaheer et al., NIPS 2017)
- End-to-End Memory Networks (Sukhbaatar et al., NIPS 2015)
- Graph Attention Networks (Veličković et al., ICLR 2018)
- Graph Attention Networks v2 (Brody et al., ICLR 2022)
- Message-Passing Neural Networks (Gilmer et al., ICML 2017)
- Pointer Graph Networks (Veličković et al., NeurIPS 2020)

If you want to implement a new processor, the easiest way is to add
it in the `processors.py` file and make it available through the
`get_processor_factory` method there. A processor should have a `__call__`
method like this:

```
__call__(self,
         node_fts, edge_fts, graph_fts,
         adj_mat, hidden,
         nb_nodes, batch_size)
```

where `node_fts`, `edge_fts` and `graph_fts` will be float arrays of shape
`batch_size` x `nb_nodes` x H, `batch_size` x `nb_nodes` x `nb_nodes` x H,
and `batch_size` x H with encoded features for
nodes, edges and graph respectively, `adj_mat` a
`batch_size` x `nb_nodes` x `nb_nodes` boolean
array of connectivity built from hints and inputs, and `hidden` a
`batch_size` x `nb_nodes` x H float array with the previous-step outputs
of the processor. The method should return a `batch_size` x `nb_nodes` x H
float array.

For more fundamentally different baselines, it is necessary to create a new
class that extends the Model API (as found within `clrs/_src/model.py`).
`clrs/_src/baselines.py` provides one example of how this can be done.

## Creating your own dataset

We provide a `tensorflow_dataset` generator class in `dataset.py`. This file can
be modified to generate different versions of the available algorithms, and it
can be built by using `tfds build` after following the installation instructions
at https://www.tensorflow.org/datasets.

Alternatively, you can generate samples without going through `tfds` by
instantiating samplers with the `build_sampler` method in
`clrs/_src/samplers.py`, like so:

```
sampler, spec = clrs.build_sampler(
    name='bfs',
    seed=42,
    num_samples=1000,
    length=16)

def _iterate_sampler(batch_size):
  while True:
    yield sampler.next(batch_size)

for feedback in _iterate_sampler(batch_size=32):
  ...

```

## Adding new algorithms

Adding a new algorithm to the task suite requires the following steps:

1. Determine the input/hint/output specification of your algorithm, and include
it within the `SPECS` dictionary of `clrs/_src/specs.py`.
2. Implement the desired algorithm in an abstractified form. Examples of this
can be found throughout the `clrs/_src/algorithms/` folder.
  - Choose appropriate moments within the algorithm’s execution to create probes
    that capture the inputs, outputs and all intermediate state (using
    the `probing.push` function).
  - Once generated, probes must be formatted using the `probing.finalize`
    method, and should be returned together with the algorithm output.
3. Implement an appropriate input data sampler for your algorithm,
and include it in the `SAMPLERS` dictionary within `clrs/_src/samplers.py`.

Once the algorithm has been added in this way, it can be accessed with the
`build_sampler` method, and will also be incorporated to the dataset if
regenerated with the generator class in `dataset.py`, as described above.

## Citation

To cite the CLRS Algorithmic Reasoning Benchmark:

```latex
@article{deepmind2022clrs,
  title={The CLRS Algorithmic Reasoning Benchmark},
  author={Petar Veli\v{c}kovi\'{c} and Adri\`{a} Puigdom\`{e}nech Badia and
    David Budden and Razvan Pascanu and Andrea Banino and Misha Dashevskiy and
    Raia Hadsell and Charles Blundell},
  journal={arXiv preprint arXiv:2205.15659},
  year={2022}
}
```
