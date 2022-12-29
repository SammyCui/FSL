from model.base import BaseModel
from utils.metric import euclidean_distance


class ProtoNet(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.temperature = args.temperature

    def _forward(self, x):

        if self.training:
            support_embd, query_embd = x[self.support_idx_train], x[self.query_idx_train]
            proto = support_embd.view(self.args.n_ways_train, self.args.n_shots_train, -1).mean(dim=1)
        else:
            support_embd, query_embd = x[self.support_idx_test], x[self.query_idx_test]
            proto = support_embd.view(self.args.n_ways_test, self.args.n_shots_test, -1).mean(dim=1)
        logits = euclidean_distance(proto=proto, query=query_embd) / self.temperature

        return logits



