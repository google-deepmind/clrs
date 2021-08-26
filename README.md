# The CLRS Algorithmic Reasoning Benchmark

Learning representations of algorithms is an emerging area of machine learning,
seeking to bridge concepts from neural networks with classical algorithms. The
CLRS Algorithmic Reasoning Benchmark (CLRS) consolidates and extends previous
work torward evaluation algorithmic reasoning by providing a suite of
implementations of classical algorithms. These algorithms have been selected
from the third edition of the standard *Introduction to Algorithms* by Cormen,
Leiserson, Rivest and Stein.

## Installation

The CLRS Algorithmic Reasoning Benchmark can be installed with pip directly from
GitHub, with the following command:

TODO(budden): Populate.

## Getting started

To set up a Python virtual environment with the required dependencies, run:

```shell
python3 -m venv clrs_env
source clrs_env/bin/activate
python setup.py install
```

and to run our example baseline model:

```shell
python -m clrs.examples.run
```

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

|       | Trajectories | Problem Size |
|-------|--------------|--------------|
| Train | 1000         | 20           |
| Eval  | 32           | 20           |
| Test  | 32           | 100          |


where "problem size" refers to e.g. the length of an array or number of nodes in
a graph, depending on the algorithm. These trajectories can be used like so:

```python
train_ds, spec = clrs.clrs21_train("bfs")

for step in range(num_train_steps):
  feedback = train_sampler.next(batch_size)
  model.train(feedback.features)
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

Please see the `examples` directory for full working Graph Neural Network (GNN)
examples using JAX and the DeepMind JAX Ecosystem of libraries.

## What we provide

### Algorithms

Our initial CLRS-21 benchmark includes the following 21 algorithms. More
algorithms will be supported in the near future.

- Divide and conquer
  - Maximum subarray (Kadane)
- Dynamic programming
  - Matrix chain order
  - Optimal binary search tree
- Graphs
  - Depth-first search
  - Breadth-first search
  - Topological sort
  - Minimum spanning tree (Prim)
  - Single-source shortest-path (Bellman Ford)
  - Single-source shortest-path (Dijsktra)
  - DAG shortest paths
  - All-pairs shortest-path (Floyd Warshall)
- Greedy
  - Task scheduling
- Searching
  - Minimum
  - Binary search
  - Quickselect
- Sorting
  - Insertion sort
  - Bubble sort
  - Heapsort
  - Quicksort
- Strings
  - String matcher (naive)
  - String matcher (KMP)

### Baselines

We additionally provide JAX implementations of the following GNN baselines:

- Graph Attention Networks (Velickovic et al., ICLR 2018)
- Message-Passing Neural Networks (Gilmer et al., ICML 2017)

## Citation

To cite the CLRS Algorithmic Reasoning Benchmark:

```latex
@article{deepmind2021clrs,
  author = {Petar Veli\v{c}kovi\'{c} and Adri\`{a} Puigdom\`{e}nech Badia and
  David Budden and Razvan Pascanu and Raia Hadsell and Charles Blundell},
  title = {The CLRS Algorithmic Reasoning Benchmark},
  year = {2021},
}
```
