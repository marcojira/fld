import torch
from fls.MoG import MoG, compute_dists, preprocess_feat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SampleEvaluator:
    def get_sample_quality_scores(self, train_feat, test_feat, gen_feat):
        """
        Gets quality score for each sample. Quality score is the dimension-adjusted LL of a generated sample from MoG centered at test samples fit to the train samples.

        A higher score (i.e. LL) would indicate a higher quality sample
        """
        train_feat, test_feat, gen_feat = preprocess_feat(
            train_feat, test_feat, gen_feat
        )

        mog_test = MoG(test_feat)
        mog_test.fit(train_feat)
        lls = mog_test.get_lls(gen_feat)
        return lls / train_feat.shape[1]

    def get_sample_memorization_scores(self, train_feat, test_feat, gen_feat):
        """
        Gets memorizations score for each sample.

        Memorization score for a generated sample is the max LL of a train sample for a Gaussian centered at the generated sample fit to the train samples.

        A higher score (i.e. LL) indicates a more memorized sample
        """
        train_feat, test_feat, gen_feat = preprocess_feat(
            train_feat, test_feat, gen_feat
        )
        mog_gen = MoG(gen_feat)
        mog_gen.fit(train_feat)

        pairwise_lls = mog_gen.get_pairwise_lls(train_feat)
        return pairwise_lls.max(dim=0).values
