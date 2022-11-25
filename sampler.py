class NegativeSampler(object):
    def __init__(self,graph,k):
        self.weights = {
            etype: graph.in_degree(etype=etype).float() ** 0.75
            for _,etype,_ in graph.canonical_etype
        }

        self.k = k
    def __call__(self, graph, eids_dict):
        result_dict = {}
        for etype,eids in eids_dict.items():
            src,_ = graph.find_edges(eids,etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinoimal(len(src),replacement=True)
            result_dict[etype] = (src,dst)
        return result_dict