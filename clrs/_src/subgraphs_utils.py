import jax.numpy as jnp


# ([B, N], [B, N, N]) -> [B, N, N]
# NOTE: also keeps edges across egonets
def get_egonets(center_nodes, adj_mat):
    """Returns the adj matrix consisting of ego nets around `center_nodes`

    Since jnp.nonzero is not compatible with JIT, we add an auxiliary node for each graph,
    and an auxiliary graph in the batch and use the `fill_value` arg to return indices to those
    added tensors.
    """
    num_graphs, num_nodes = center_nodes.shape

    # Add one node and one graph
    center_nodes = jnp.concatenate([jnp.zeros((1, num_nodes)), center_nodes])
    center_nodes = jnp.concatenate([jnp.zeros((num_graphs + 1, 1)), center_nodes], axis=-1)

    adj_mat = jnp.concatenate([jnp.zeros((1, num_nodes, num_nodes)), adj_mat])
    adj_mat = jnp.concatenate([jnp.zeros((num_graphs + 1, 1, num_nodes)), adj_mat], axis=1)
    adj_mat = jnp.concatenate([jnp.zeros((num_graphs + 1, num_nodes + 1, 1)), adj_mat], axis=-1)

    graph_idx, node_idx = center_nodes.nonzero(size=(num_graphs+1)*(num_nodes+1), fill_value=0)

    # [K, N] where K is the total number of center_nodes (summed over graphs)
    center_adj_cols = adj_mat[graph_idx, :, node_idx]

    # [B, N]: for each graph, whether node n is a neighbour of a center_node
    center_neighbors = jnp.zeros((adj_mat.shape[0], adj_mat.shape[-1]))
    center_neighbors = center_neighbors.at[graph_idx].add(center_adj_cols)

    # Add center nodes
    ego_nodes = center_neighbors + center_nodes

    graph_idx, removed_node_idx = (ego_nodes == 0).nonzero(size=(num_graphs+1)*(num_nodes+1), fill_value=0)

    # Zero out edges incoming/outgoing to/from removed nodes
    adj_mat = adj_mat.at[graph_idx, removed_node_idx].set(0)
    adj_mat = adj_mat.at[graph_idx, :, removed_node_idx].set(0)

    # Remove the added node and graph
    adj_mat = adj_mat[1:, 1:, 1:]
    return adj_mat

# ([B, N], [B, N, N]) -> [B, N, N]
def get_stars(center_nodes, adj_mat):
    """Returns the adj matrix consisting of star subgraphs around `center_nodes`

    Since jnp.nonzero is not compatible with JIT, we add an auxiliary node for each graph,
    and an auxiliary graph in the batch and use the `fill_value` arg to return indices to those
    added tensors.
    """
    num_graphs, num_nodes = center_nodes.shape

    # Add one node and one graph
    center_nodes = jnp.concatenate([jnp.zeros((1, num_nodes)), center_nodes])
    center_nodes = jnp.concatenate([jnp.zeros((num_graphs + 1, 1)), center_nodes], axis=-1)

    adj_mat = jnp.concatenate([jnp.zeros((1, num_nodes, num_nodes)), adj_mat])
    adj_mat = jnp.concatenate([jnp.zeros((num_graphs + 1, 1, num_nodes)), adj_mat], axis=1)
    adj_mat = jnp.concatenate([jnp.zeros((num_graphs + 1, num_nodes + 1, 1)), adj_mat], axis=-1)

    graph_idx, node_idx = center_nodes.nonzero(size=(num_graphs+1)*(num_nodes+1), fill_value=0)

    # [K, N] where K is the total number of center_nodes (summed over graphs)
    center_adj_cols = adj_mat[graph_idx, :, node_idx]
    center_adj_rows = adj_mat[graph_idx, node_idx, :]

    # Zero out all edges, except those incoming/outgoing to/from center_nodes
    new_adj_mat = jnp.zeros(adj_mat.shape)
    new_adj_mat = new_adj_mat.at[graph_idx, :, node_idx].set(center_adj_cols)
    new_adj_mat = new_adj_mat.at[graph_idx, node_idx].add(center_adj_rows)

    # Remove the added node and graph
    new_adj_mat = new_adj_mat[1:, 1:, 1:]
    return new_adj_mat
