""" PyTorch reimplementation from https://github.com/kynkaat/improved-precision-and-recall-metric """
import torch
from fls.metrics.Metric import Metric


class PrecisionRecall(Metric):
    # We use 4 to ignore distance to self
    def __init__(self, mode, num_neighbors=4):
        super().__init__()
        self.name = mode
        self.mode = mode
        self.num_neighbors = num_neighbors

    def get_nn_dists(self, feat):
        dists = torch.cdist(feat, feat)
        dists = dists.topk(self.num_neighbors, dim=1, largest=False).values
        return dists[:, -1]

    def pct_in_manifold(self, evaluated_feat, manifold_feat):
        nn_dists = self.get_nn_dists(manifold_feat)
        pairwise_dists = torch.cdist(evaluated_feat, manifold_feat)

        comparison_tensor = nn_dists.unsqueeze(0).repeat(pairwise_dists.shape[0], 1)

        num_in_manifold = (pairwise_dists < comparison_tensor).sum(dim=1)
        num_in_manifold = (num_in_manifold > 0).sum()
        return num_in_manifold / evaluated_feat.shape[0]

    def compute_metric(
        self,
        train_feat,
        test_feat,  # Test samples not used by Precision/Recall
        gen_feat,
    ):
        train_feat = train_feat.cuda()
        gen_feat = gen_feat.cuda()

        if self.mode == "Precision":
            return self.pct_in_manifold(gen_feat, train_feat).cpu().item()
        elif self.mode == "Recall":
            return self.pct_in_manifold(train_feat, gen_feat).cpu().item()
        else:
            raise NotImplementedError
