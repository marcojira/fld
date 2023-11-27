from fls.metrics.AuthPct import AuthPct
from fls.metrics.CTTest import CTTest
from fls.metrics.FID import FID
from fls.metrics.FLS import FLS
from fls.metrics.KID import KID
from fls.metrics.PrecisionRecall import PrecisionRecall

# For testing
ALL_METRICS = [
    AuthPct(),
    CTTest(),
    FID("train"),
    FID("test"),
    FLS("train"),
    FLS("test"),
    KID("train"),
    KID("test"),
    PrecisionRecall("Precision"),
    PrecisionRecall("Recall"),
]
ALL_METRICS_DICT = {metric.name: metric for metric in ALL_METRICS}
