from fld.metrics.AuthPct import AuthPct
from fld.metrics.CTTest import CTTest
from fld.metrics.FID import FID
from fld.metrics.FLD import FLD
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall

# For testing
ALL_METRICS = [
    AuthPct(),
    CTTest(),
    FID("train"),
    FID("test"),
    FLD("train"),
    FLD("test"),
    KID("train"),
    KID("test"),
    PrecisionRecall("Precision"),
    PrecisionRecall("Recall"),
]
ALL_METRICS_DICT = {metric.name: metric for metric in ALL_METRICS}
