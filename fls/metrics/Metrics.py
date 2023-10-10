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
    FID("test", ref_size=4000),
    FLS("test"),
    FLS("train"),
    KID("train"),
    KID("test"),
    PrecisionRecall("Precision"),
    PrecisionRecall("Recall"),
]

MAIN_METRICS = [
    # AuthPct(),
    # CTTest(),
    # FID(),
    FLS(),
    KID(),
    PrecisionRecall("Precision"),
    PrecisionRecall("Recall"),
]
