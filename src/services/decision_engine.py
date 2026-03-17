# src/services/decision_engine.py

import numpy as np
from sklearn.metrics import roc_curve

class DecisionEngine:

    APPROVE = "APPROVE"
    FLAG    = "FLAG"
    BLOCK   = "BLOCK"

    def __init__(self, threshold_approve: float = 0.1888,
                       threshold_block: float  = 0.6973):
        self.threshold_approve = threshold_approve
        self.threshold_block   = threshold_block

    def decide(self, risk_score: float) -> str:
        if risk_score < self.threshold_approve:
            return self.APPROVE
        elif risk_score < self.threshold_block:
            return self.FLAG
        else:
            return self.BLOCK

    def set_thresholds(self, approve: float, block: float) -> None:
        assert approve < block, "threshold_approve must be less than threshold_block"
        self.threshold_approve = approve
        self.threshold_block   = block

    def tune_thresholds(self, y_true, y_scores,
                        target_fpr: float = 0.02,
                        approve_contamination: float = 0.005) -> tuple:
        """
        Derives both thresholds from validation data.
        Returns (threshold_approve, threshold_block) and sets them on self.
        """
        thresholds = np.linspace(0.001, 0.999, 1000)
        results = []

        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            tn = ((preds == 0) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()

            fpr                  = fp / (fp + tn) if (fp + tn) > 0 else 0
            approved_fraud_rate  = fn / (tn + fn) if (tn + fn) > 0 else 0
            results.append((t, fpr, approved_fraud_rate))

        # threshold_block: lowest t where FPR < target_fpr
        block_candidates = [(t, fpr, afr) for t, fpr, afr in results if fpr < target_fpr]
        threshold_block  = block_candidates[0][0]

        # threshold_approve: highest t where approved fraud rate < contamination cap
        approve_candidates = [(t, fpr, afr) for t, fpr, afr in results
                              if afr < approve_contamination]
        threshold_approve  = approve_candidates[-1][0]

        self.set_thresholds(threshold_approve, threshold_block)
        return threshold_approve, threshold_block
