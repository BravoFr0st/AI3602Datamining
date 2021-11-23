from header import Community

MAX_EDGES = 220377
def nodewise_delta_q(node: int, community: Community):
    graph = community.G
    intra_in_deg = float(community.intra_comm_in_degree(node))
    intra_out_deg = float(community.intra_comm_out_degree(node))
    # intra_in_deg = 10
    # intra_out_deg = 10
    in_deg = float(graph.get_in_degree(node))
    out_deg = float(graph.get_out_degree(node))
    comm_out_deg = float(community.out_deg)
    comm_in_deg = float(community.in_deg)
    return (intra_in_deg + intra_out_deg) / graph.sumedge -\
        (in_deg * comm_out_deg + out_deg * comm_in_deg) / (graph.sumedge ** 2)

def commwise_delta_q(comm1: Community, comm2: Community):
    dq_loss = (comm1.in_deg * comm2.out_deg + comm1.out_deg * comm2.in_deg)
    dq_loss = dq_loss / MAX_EDGES / MAX_EDGES
    dq_gain = 0
    for node in comm2.nodes:
        dq_gain += comm1.intra_comm_out_degree(node)
        dq_gain += comm1.intra_comm_in_degree(node)

    dq_gain = dq_gain / MAX_EDGES

    return dq_gain - dq_loss
