import graphlib as gl
import networkit as nt

# cyclic graph to test
cyclic = [2,0,1]
acyclic = [-1, 0]
def is_acyclic(input, pi):
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
    graph = nt.graph.Graph(n = len(pi), directed = True)
    for i in range(len(pi)):
        for j in range(len(pi)):
            if input[i,j] == 1:
                graph.addEdge(i,j)
    # no self-loop on the start node
    pi[0] = -1

    # check self-looping conditions
    if is_valid_self_loops(input, pi):
        for i in range(len(pi)):
            if pi[i] == i:
                pi[i] = -1
    else:
        return False





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


def is_valid_self_loops(input, pi):
    """
    Checks whether self-loops stem from valid DFS execution.
    :param input:
    :param pi:
    :return:
    """
    for i in range(len(pi)):
        if pi[i] == i:
            for j in range(i):
                paths = nt.reachability.AllSimplePaths(input, j, i)
                simple_paths = paths.numberOfSimplePaths()
                if simple_paths > 0:
                    return False
                else:
                    pi[i] = -1
    return True

print(is_acyclic(acyclic))