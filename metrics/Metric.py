class Metric:
    """Generic Metric class"""

    def __init__(self):
        pass

    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
    ):
        """Computes the metric value for the given sets of features (TO BE IMPLEMENTED BY EACH METRIC)

        - train_feat: Features from set of samples used to train generative model
        - baseline_feat: Features from some baseline set of data belonging to the same distribution as train_feat/test_feat
            - Generally taken by splitting the training set in two, using the first half for train_feat and the second for baseline_feat
                - Could also be taken by splitting the test set.
            - IMPORTANT: baseline_feat and train_feat/test_feat should be MUTUALLY EXCLUSIVE to avoid skewing the results
        - test_feat: Features from test samples
        - gen_feat: Features from generated samples

        returns: Metric value
        """
        pass

    def create_metric_dict(
        self, train_feat, baseline_feat, test_feat, gen_feat_dict, plot=False
    ):
        """Returns a dictionary of gen_name -> metric value for each set of generated features in gen_feat_dict"""
        return {
            gen_name: self.compute_metric(
                train_feat, baseline_feat, test_feat, gen_feat, plot=plot
            )
            for gen_name, gen_feat in gen_feat_dict.items()
        }
