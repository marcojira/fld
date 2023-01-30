class Metric:
    def __init__(self):
        pass

    def compute_metric(
        self,
        train_feat,
        baseline_feat,
        test_feat,
        gen_feat,
        plot=False,
    ):
        pass

    def create_metric_dict(
        self, train_feat, baseline_feat, test_feat, gen_feat_dict, plot=False
    ):
        return {
            metric_name: self.compute_metric(
                train_feat, baseline_feat, test_feat, gen_feat, plot=plot
            )
            for metric_name, gen_feat in gen_feat_dict.items()
        }
