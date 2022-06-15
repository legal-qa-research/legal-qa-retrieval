import pickle

from legal_graph.graph_entities.legal_graph import LegalGraph
from utils.constant import pkl_legal_graph

if __name__ == '__main__':
    legal_graph: LegalGraph = pickle.load(open(pkl_legal_graph, 'rb'))
    max_neighbor = 0
    total_ref = 0
    for node in legal_graph.lis_node:
        n_neighbor = len(node.lis_neighbor)
        max_neighbor = max(max_neighbor, n_neighbor)
        total_ref += n_neighbor
    print('Max neighbor: ', max_neighbor)
    print('Mean neighbor: ', total_ref / len(legal_graph.lis_node))
