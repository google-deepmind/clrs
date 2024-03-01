import graphlib as gl
import networkx as nx
import numpy as np

## f[0][0][1].data to get adjacency matrix from next(sampler) where sampler=test_samplers[0]

# cyclic graph to test
cyclic_adj = np.array([
    [0,1,0],
    [0,0,1],
    [1,0,0]
])
cyclic_pi = [2,0,1]

acyclic_adj = np.array([
    [0,1],
    [0,0]
])
acyclic_pi = [0, 0]

def is_acyclic(np_input_array, pi):
    """
    Function to check for cycles in a predecessor array returned by the model
    :param input: the adjacency matrix of the graph on which the model has inferred the predecessor array
    :param pi: the predecessor array
    :return: Boolean indicating acyclicity
    """

    # if self-loop: is i reachable from a lower-indexed node?
        # if yes: return false
        # if no: replace its parent by the god node -1

    # Build networkit graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(pi)))
    for i in range(len(pi)):
        for j in range(len(pi)):
            if np_input_array[i][j] == 1:
                graph.add_edge(i,j)
    # no self-loop on the start node
    #pi[0] = -1

    # check self-looping conditions
    if is_valid_self_loops(np_input_array, pi):
        for i in range(len(pi)):
            if pi[i] == i:
                pi[i] = -1
    else:
        return False

    print(pi)




    breakpoint()
    ts = gl.TopologicalSorter()
    for i in range(len(pi)):
        ts.add(i, pi[i])
    try:
        ts.prepare()
        return True
    except ValueError as e:
        if isinstance(e, gl.CycleError):
            print("I am a cycle error")
            return False
        else:
            raise e


def is_valid_self_loops(np_input_array, pi):
    """
    Checks whether self-loops stem from valid DFS execution.
    :param input:
    :param pi:
    :return:
    """
    g = nx.from_numpy_array(np_input_array)
    for i in range(len(pi)):
        if pi[i] == i:
            for j in range(i):
                reachable_by_earlier_node = nx.has_path(g, j, i)
                if reachable_by_earlier_node:
                    return False
    return True

print(is_acyclic(acyclic_adj, acyclic_pi))
print(is_acyclic(cyclic_adj, cyclic_pi))
