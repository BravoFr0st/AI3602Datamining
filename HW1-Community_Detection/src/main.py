from collections import Counter
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
import scipy.sparse as sparse
from collections import defaultdict

MAX_NODES = 31136
n_nodes = 31136
MAX_EDGES = 220377

EDGE_PATH = './data/edges_update.csv'
GT_PATH='./data/ground_truth.csv'
RESULT_PATH = './res.csv'
class Graph():
    def __init__(self,adj_mat: sparse.coo_matrix,):
        self.coo = adj_mat
        self.out_degrees = self.coo.sum(axis=1).A1
        self.in_degrees = self.coo.sum(axis=0).A1

        self.neighbours= defaultdict(set)
        self.in_neighbours= defaultdict(set)
        self.out_neighbours= defaultdict(set)
        self.edge_weights = {} # key =tuple[int,int]  value=int
        
        for src, dst, weight in zip(self.coo.row, self.coo.col, self.coo.data):
            self.neighbours[src].add(dst)
            self.neighbours[dst].add(src)
            self.in_neighbours[dst].add(src)
            self.out_neighbours[src].add(dst)
            self.edge_weights[(src, dst)] = weight
        self.sumedge = float(self.coo.sum())

    def set_node_to_comm(self, node2comm):
        self.node2comm = node2comm

    def get_in_degree(self, node: int):
        return self.in_degrees[node]

    def get_out_degree(self, node: int):
        return self.out_degrees[node]

    def get_neighbours(self, node: int):
        return self.neighbours[node]

    def get_in_neighbours(self, node: int):
        return self.in_neighbours[node]

    def get_out_neighbours(self, node: int):
        return self.out_neighbours[node]

class Community():
    def __init__(self, graph: Graph):
        self.nodes= set([])
        self.G = graph
        self.out_deg: int = 0
        self.in_deg: int = 0

    def add_node(self, node: int):
        self.nodes.add(node)
        self.out_deg += self.G.out_degrees[node]
        self.in_deg += self.G.in_degrees[node]

    def remove_node(self, node: int):
        self.nodes.remove(node)
        self.out_deg -= self.G.out_degrees[node]
        self.in_deg -= self.G.in_degrees[node]

    def intra_comm_in_degree(self, node: int):
        in_deg = 0.
        in_neighbours = self.G.get_in_neighbours(node)
        for neighbour in in_neighbours:
            if neighbour in self.nodes:
                in_deg += self.G.edge_weights[(neighbour, node)]

        return in_deg

    def intra_comm_out_degree(self, node: int):
        out_deg = 0.
        out_neighbours = self.G.get_out_neighbours(node)
        for neighbour in out_neighbours:
            if neighbour in self.nodes:
                out_deg += self.G.edge_weights[(node, neighbour)]

        return out_deg

edges = pd.read_csv(EDGE_PATH).to_numpy()
ground_truth=pd.read_csv(GT_PATH).to_numpy()

graph_coo = sparse.coo_matrix(
    (np.ones(edges.shape[0]), (edges.T[0], edges.T[1])),
)
graph_coo.sum_duplicates()

def nodewise_delta_q(node, community):
    graph = community.G
    intra_in_deg = float(community.intra_comm_in_degree(node))
    intra_out_deg = float(community.intra_comm_out_degree(node))

    in_deg = float(graph.get_in_degree(node))
    out_deg = float(graph.get_out_degree(node))
    comm_out_deg = float(community.out_deg)
    comm_in_deg = float(community.in_deg)
    return (intra_in_deg + intra_out_deg) / graph.sumedge -\
        (in_deg * comm_out_deg + out_deg * comm_in_deg) / (graph.sumedge ** 2)


def commwise_delta_q(comm1, comm2):
    dq_gain = 0
    for node in comm2.nodes:
        dq_gain += comm1.intra_comm_out_degree(node)+comm1.intra_comm_in_degree(node)+dq_gain
    return dq_gain / MAX_EDGES - (comm1.in_deg * comm2.out_deg + comm1.out_deg * comm2.in_deg)/ MAX_EDGES / MAX_EDGES

def graph_reindex(graph: Graph):
    reindexer = {}
    comms = set(graph.node2comm)
    for idx, comm in enumerate(comms):
        reindexer[comm] = idx
    return reindexer


def rebuild_metagraph_coo(old_graph: Graph, n_nodes: int):
    reindexer = graph_reindex(old_graph)
    new_src = []
    new_dst = []
    edge_weights = []
    for node in range(n_nodes):
        comm_src = metagraph.node2comm[node]
        comm_src_ridx = reindexer[comm_src]
        for neighbour in metagraph.get_out_neighbours(node):
            comm_dst = metagraph.node2comm[neighbour]
            comm_dst_ridx = reindexer[comm_dst]
            new_src.append(comm_src_ridx)
            new_dst.append(comm_dst_ridx)
            edge_weights.append(metagraph.edge_weights[(node, neighbour)])
    new_matrix = sparse.coo_matrix((edge_weights, (new_src, new_dst)))
    new_matrix.sum_duplicates()

    return new_matrix


def merge_community(a,b,communities):
    comm1 = communities[a]
    comm2 = communities[b]
    for node in comm2.nodes.copy():
        comm1.add_node(node)
        comm2.remove_node(node)
        global_node2comm[node] = a


def refresh_communities(
        global_node2comm,
        GRAPH):
    community_counter = Counter(global_node2comm)
    communities = []
    for i in range(len(community_counter)):
        communities.append(Community(GRAPH))
    for node in range(MAX_NODES):
        comm = global_node2comm[node]
        communities[comm].add_node(node)

    return communities


# init partition
metagraph = Graph(graph_coo)
GRAPH = Graph(graph_coo)
global_node2comm = []
node2comm = []
communities = []

for node in range(n_nodes):
    global_node2comm.append(node)
    node2comm.append(node)
    community = Community(metagraph)
    community.add_node(node)
    communities.append(community)

metagraph.set_node_to_comm(node2comm)

num_iter = 0

while True:
    modularity = 0
    #unchanged = False
    while True:
        unchanged = True
        num_iter += 1
        print('Iteration '+str(num_iter)+':')
        num_nodes_changed = 0
        rndlist=list(range(n_nodes))
        random.shuffle(rndlist)

        for node in tqdm(rndlist):

            max_delta_Q = 0

            old_comm_idx = metagraph.node2comm[node]
            old_comm = communities[old_comm_idx]
            old_comm.remove_node(node)

            best_comm_idx = old_comm_idx
            best_comm = old_comm
            metagraph.node2comm[node] = -1
            
            for neighbour in metagraph.get_neighbours(node):
                new_comm_idx = metagraph.node2comm[neighbour]
                #if new_comm_idx == old_comm_idx:
                #    continue
                new_comm = communities[new_comm_idx]
                delta_q = nodewise_delta_q(node, new_comm)-nodewise_delta_q(node, old_comm)

                if delta_q > max_delta_Q: #update
                    unchanged = False
                    max_delta_Q = delta_q
                    best_comm = new_comm
                    best_comm_idx = new_comm_idx

            metagraph.node2comm[node] = best_comm_idx
            modularity += max_delta_Q

            best_comm.add_node(node)
            if old_comm_idx != best_comm_idx:
                num_nodes_changed += 1

        print('Modularity: '+str(modularity))
        print('Changed: '+str(num_nodes_changed))
        print('Communities: '+str(len(set(metagraph.node2comm))))

        if unchanged == True:
            break

    # reindex
    reindexer=graph_reindex(metagraph)
    for node in range(MAX_NODES):
        metanode=global_node2comm[node]  
        new_metanode=metagraph.node2comm[metanode]  
        new_metanode_ridx=reindexer[new_metanode]  
        global_node2comm[node]=new_metanode_ridx
    GRAPH.set_node_to_comm(global_node2comm)

    # construct new metagraph
    new_metagraph=rebuild_metagraph_coo(metagraph, n_nodes)

    if graph_coo.shape == new_metagraph.shape:
        if (graph_coo - new_metagraph).sum() == 0:
            print('No new change')
            break

    graph_coo=new_metagraph

    metagraph=Graph(graph_coo)
    node2comm=[]
    communities=[]

    n_nodes=graph_coo.shape[0]

    for node in range(n_nodes):
        node2comm.append(node)
        community=Community(metagraph)
        community.add_node(node)
        communities.append(community)

    metagraph.set_node_to_comm(node2comm)


##重构
community_counter=Counter(global_node2comm)
communities=[]

for i in range(len(community_counter)):
    communities.append(Community(GRAPH))
for node in range(MAX_NODES):
    comm=global_node2comm[node]
    communities[comm].add_node(node)

while len(community_counter) > 5:
    sorted_counter=sorted(community_counter.items(), key=lambda x: x[1])
    candidates=sorted_counter[:2]
    merge_community(candidates[0][0], candidates[1][0], communities)
    community_counter=Counter(global_node2comm)



with open(RESULT_PATH, 'w') as f:
    f.write('id, category\n')
    for id, category in enumerate(global_node2comm):
        f.write(f'{id}, {graph_reindex(GRAPH)[category]}\n')

result=pd.read_csv(RESULT_PATH).to_numpy()

def evaluation():
    all_rangelist = [list(range(5))]
    
    while len(all_rangelist)<120: #C_5^5
        rangelist=list(range(5))
        random.shuffle(rangelist)
        if rangelist not in all_rangelist:
            all_rangelist.append(rangelist)
    
    #只要遍历到所有排序就行

    for i, j in result:
        assert result[i, 0] == i

    max_rate = 0.0
    for it in all_rangelist: 
        vote = 0
        for i, j in ground_truth:
            if result[i, 1] == it[j]:
                vote += 1
        max_rate = max(max_rate, vote / len(ground_truth))
    print(max_rate)

print('evaluating')
evaluation() 
