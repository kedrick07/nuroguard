"""
src/services/explanation_service.py
NuroGuard — SHAP Explainability Layer

Converts a raw feature vector into top-3 plain-language reason codes
that explain WHY a transaction was Approved / Flagged / Blocked.

Dependencies:
    shap       — TreeExplainer + SHAP value extraction
    numpy      — absolute value sorting
    lightgbm   — Booster type check
    pickle     — model loading
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
# Add this directly below the existing imports block
import warnings
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed",
    category=UserWarning,
    module="shap",
)

class ExplanationService:
    """
    Wraps a trained LightGBM model in a SHAP TreeExplainer and converts
    feature-level attributions into plain-language reason codes.

    Reason codes are designed for low-literacy users (ASEAN unbanked persona):
    plain English, no jargon, no raw feature names visible to the end user.
    """

    # ------------------------------------------------------------------ #
    # Reason Code Templates                                                #
    # Each key is a feature name produced by FeatureEngineer.             #
    # {multiplier} is filled in at runtime using the actual feature value. #
    # ------------------------------------------------------------------ #
    REASON_TEMPLATES: dict[str, str] = {
        # --- v2.0 UID / group aggregation features ---
        "uid_amt_deviation":       "Amount is {multiplier}× higher than your usual spending pattern",
        "uid_amt_zscore":          "Amount deviates {multiplier} standard deviations from your norm",
        "card1_amt_deviation":     "This amount is unusual compared to your card's history",
        "card1_amt_zscore":        "Spending level is far outside your card's normal range",
        "uid2_amt_deviation":      "Amount is {multiplier}× higher than your typical transaction",
        "uid_txn_count":           "Unusually high number of transactions from this account",

        # --- v2.0 Amount decomposition ---
        "is_round_amount":         "Suspiciously round amount — a common pattern in card-testing fraud",
        "amt_decimal":             "Exact decimal value matches known card-testing amounts",
        "log_amt":                 "Transaction amount is unusually large",

        # --- v2.0 Null-as-signal flags ---
        "id_30_was_null":          "Transaction came from an unrecognised operating system",
        "id_31_was_null":          "Transaction came from an unrecognised browser",
        "id_33_was_null":          "Screen resolution data is missing — unrecognised device",
        "DeviceType_was_null":     "Device type could not be identified",
        "DeviceInfo_was_null":     "Device information is missing — possible spoofed device",

        # --- Velocity / time features ---
        "velocity_ratio":          "You are making transactions {multiplier}× faster than usual",
        "hour":                    "Transaction at an unusual hour for your account",
        "is_night":                "Late-night transaction — outside your normal activity window",
        "is_weekend":              "Weekend transaction pattern differs from your history",
        "day_of_week":             "This day-of-week shows elevated fraud in your pattern",

        # --- Geo-velocity ---
        "implied_speed_kmh":       "Transaction location implies physically impossible travel speed",
        "is_impossible_travel":    "You cannot have travelled from your last known location this quickly",
        "is_new_country":          "Transaction from a country not previously seen on your account",
        "distance_km":             "Transaction is far from your most recent location",

        # --- Behavioural features ---
        "amount_ratio":            "Amount is {multiplier}× your average transaction",
        "is_new_device":           "Transaction from a device you have not used before",
        "is_unusual_hour":         "Transaction time is outside your typical activity hours",

        # --- Raw IEEE-CIS identity columns ---
        "id_02":                   "Identity verification score is unusually high",
        "id_05":                   "Email address linked to recent chargebacks",
        "id_06":                   "Billing address mismatch detected",
        "TransactionAmt":          "Transaction amount is a strong fraud indicator for this pattern",
        "card1":                   "Card number pattern matches flagged transaction clusters",
        "addr1":                   "Billing address is linked to elevated fraud activity",
        "P_emaildomain":           "Email domain is associated with elevated fraud risk",
        "C1":                      "Number of payment addresses on this card is elevated",
        "C13":                     "High count of unique billing addresses — possible account sharing",
        "D1":                      "Days since last transaction is unusually short",
        "D15":                     "Account age indicates a newly created profile",
        "V258":                    "Vesta verification score indicates elevated risk",
        "V308":                    "Advanced verification signal flagged this transaction",
    
        # --- Raw IEEE-CIS card columns ---
        "card1":    "Card number pattern matches flagged transaction clusters",
        "card2":    "Card sub-type is associated with elevated fraud activity",
        "card3":    "Card issuer country shows elevated risk for this transaction",
        "card4":    "Card network (Visa/MC/Amex) pattern is unusual for this merchant",
        "card5":    "Card product type is linked to this fraud pattern",
        "card6":    "Card category (debit/credit/charge) is flagged for this amount",

        # --- Raw IEEE-CIS C (count) columns ---
        "C1":       "Number of payment addresses on this card is elevated",
        "C2":       "Number of unique cards used at this email is elevated",
        "C4":       "High transaction count linked to this device",
        "C5":       "Number of accounts linked to this address is elevated",
        "C6":       "Number of unique billing addresses on this account is high",
        "C7":       "Number of unique email addresses linked to this card is elevated",
        "C8":       "Number of identity mismatches on this account is elevated",
        "C9":       "Number of linked accounts shows unusual sharing pattern",
        "C10":      "Cross-account linkage count is outside normal range",
        "C11":      "Number of associated devices on this account is elevated",
        "C12":      "Number of unique shipping addresses linked to this card is high",
        "C13":      "High count of unique billing addresses — possible account sharing",
        "C14":      "Number of linked bank accounts is unusually high",

        # --- Raw IEEE-CIS D (time delta) columns ---
        "D1":       "Days since last transaction is unusually short",
        "D2":       "Time gap between transactions is outside your normal pattern",
        "D3":       "Days since last address change is very recent",
        "D4":       "Time since last card use is flagged for this pattern",
        "D5":       "Account activity gap is inconsistent with your history",
        "D10":      "Days between linked account creation is suspiciously short",
        "D15":      "Account age indicates a newly created profile",

        # --- Raw IEEE-CIS id_ columns ---
        "id_01":    "Identity score indicates elevated transaction risk",
        "id_02":    "Identity verification score is unusually high",
        "id_03":    "Device-to-account linkage score is flagged",
        "id_04":    "Session-level identity signal is outside normal range",
        "id_05":    "Email address linked to recent chargebacks",
        "id_06":    "Billing address mismatch detected",
        "id_07":    "Shipping address mismatch score is elevated",
        "id_08":    "Billing-to-shipping distance is unusually large",
        "id_09":    "IP address is inconsistent with your registered location",
        "id_10":    "IP address change detected since last transaction",
        "id_11":    "Account tenure score indicates a new or synthetic account",
        "id_13":    "Number of active sessions is higher than expected",
        "id_14":    "Timezone offset does not match your usual location",
        "id_17":    "Proxy or VPN usage detected on this transaction",
        "id_19":    "Shipping email domain differs from billing email domain",
        "id_20":    "Billing email domain is associated with elevated risk",

        # --- Raw IEEE-CIS M (match) columns ---
        "M1":       "Name on card does not match billing address record",
        "M2":       "Card name mismatch detected",
        "M3":       "Address verification returned a mismatch",
        "M4":       "Method-of-payment signal is flagged",
        "M5":       "Vesta verification returned a negative match signal",
        "M6":       "Address line 2 verification failed",
        "M7":       "Card issuer verification returned a mismatch",
        "M8":       "Shipping address verification returned a mismatch",
        "M9":       "Billing agreement verification is flagged",

        # --- Raw IEEE-CIS V (Vesta engineered) columns ---
        "V95":      "Vesta count signal indicates unusual transaction frequency",
        "V130":     "Vesta amount signal is outside your normal range",
        "V258":     "Vesta verification score indicates elevated risk",
        "V308":     "Advanced verification signal flagged this transaction",
        "V317":     "Vesta cross-account signal is elevated",
    }

    # Generic fallback — used when a feature has no matching template
    _FALLBACK_TEMPLATE = "Unusual value detected for transaction signal '{feature}'"

    # ------------------------------------------------------------------ #
    # Constructor                                                          #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model_path: Union[str, Path],
        top_n: int = 3,
    ) -> None:
        self.top_n = top_n
        self.model_path = Path(model_path)

        with open(self.model_path, "rb") as f:
            loaded = pickle.load(f)

        # --- Unpack all three possible pickle formats ---
        if isinstance(loaded, lgb.Booster):
            self._booster = loaded
            self._stored_feature_cols = None

        elif isinstance(loaded, dict):
            for key in ("model", "booster", "lgbm", "classifier", "lgbm_model"):
                if key in loaded and isinstance(loaded[key], lgb.Booster):
                    self._booster = loaded[key]
                    break
            else:
                raise KeyError(
                    f"No lgb.Booster found in pickle dict. "
                    f"Keys present: {list(loaded.keys())}"
                )
            self._stored_feature_cols = loaded.get("feature_cols", None)

        elif hasattr(loaded, "booster_"):
            self._booster = loaded.booster_
            self._stored_feature_cols = None

        else:
            raise TypeError(
                f"Unsupported pickle type: {type(loaded)}. "
                "Expected lgb.Booster, dict with Booster, or LGBMClassifier."
            )

        # --- Build definitive feature name list ---
        if self._stored_feature_cols is not None:
            self.feature_names = list(self._stored_feature_cols)
        else:
            self.feature_names = self._booster.feature_name()

        # --- Initialise SHAP TreeExplainer ---
        self.explainer = shap.TreeExplainer(self._booster)



    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def explain(self, features: pd.Series) -> list[str]:
        """
        Main entry point called by RiskScoringService.

        Given a feature vector for ONE transaction, return up to top_n
        plain-language reason codes explaining the model's decision.

        Args:
            features: pd.Series — index = feature names, values = feature values.
                      Does NOT need to be pre-sorted; we align it internally.

        Returns:
            List[str] of human-readable reason codes, ordered by importance.

        Example output:
            [
              "Amount is 4.2× higher than your usual spending pattern",
              "Suspiciously round amount — a common pattern in card-testing fraud",
              "Transaction from a device you have not used before"
            ]
        """
        shap_values = self._get_shap_values(features)
        top_feature_names = self._top_features(shap_values)

        reasons = []
        for feat_name in top_feature_names:
            feat_value = features.get(feat_name, None)
            shap_val = shap_values[self.feature_names.index(feat_name)]
            reason = self._format_reason(feat_name, feat_value, shap_val)
            reasons.append(reason)

        return reasons

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_shap_values(self, features: pd.Series) -> np.ndarray:
        """
        Compute SHAP values for a single transaction row.

        Args:
            features: pd.Series with feature names as index.

        Returns:
            1-D numpy array, shape (n_features,).
            Positive value → pushes the score TOWARD fraud.
            Negative value → pushes the score AWAY from fraud.
        """
        # Reindex to EXACTLY match the model's expected column order.
        # Any column the model knows about but our input doesn't have
        # gets filled with 0.0 as a safe neutral placeholder.
        aligned = features.reindex(self.feature_names, fill_value=0.0)

        # TreeExplainer expects shape (n_samples, n_features) — i.e., 2-D.
        # We have one row, so reshape from (n_features,) → (1, n_features).
        row_2d = aligned.values.reshape(1, -1)

        raw = self.explainer.shap_values(row_2d)

        # LightGBM binary classification returns a LIST of two arrays:
        #   raw[0] → SHAP values attributing toward class 0 (legit)
        #   raw[1] → SHAP values attributing toward class 1 (fraud)  ← we want this
        # Some SHAP versions return a single 2-D array instead of a list.
        # We handle both cases explicitly.
        if isinstance(raw, list):
            return raw[1][0]    # class 1 (fraud), sample index 0
        else:
            return raw[0]       # single array, sample index 0

    def _top_features(self, shap_values: np.ndarray) -> list[str]:
        """
        Identify the top-N features by absolute SHAP magnitude.

        We rank by |SHAP|, not raw SHAP, because:
        - A large positive SHAP → strong push TOWARD fraud
        - A large negative SHAP → strong push AWAY from fraud
        Both are equally informative for explaining a decision.

        Args:
            shap_values: 1-D array of SHAP values, shape (n_features,).

        Returns:
            List of feature name strings, ordered highest → lowest |SHAP|.
        """
        abs_shap = np.abs(shap_values)

        # argsort returns indices from lowest to highest value.
        # [::-1] reverses the order so we get highest → lowest.
        sorted_indices = np.argsort(abs_shap)[::-1]

        # Take only the top_n indices and map each back to its feature name.
        top_indices = sorted_indices[: self.top_n]
        return [self.feature_names[i] for i in top_indices]

    def _format_reason(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float,
    ) -> str:
        """
        Convert a raw (feature_name, value, shap) triplet into a plain sentence.

        Lookup order:
          1. Exact match in REASON_TEMPLATES
          2. Prefix match  (handles card2_amt_zscore, uid2_amt_deviation, etc.)
          3. Generic fallback

        Then fill in the {multiplier} placeholder if the template uses it.

        Args:
            feature_name:  e.g. 'uid_amt_deviation'
            feature_value: The actual numeric value for this transaction.
            shap_value:    SHAP attribution value — used as multiplier fallback.

        Returns:
            Plain-English string for display to the user.
        """
        # Step 1 — exact match
        template = self.REASON_TEMPLATES.get(feature_name)

        # Step 2 — prefix match
        # Handles cases like 'card2_amt_zscore' matching 'card1_amt_zscore' template
        # or any feature with a shared semantic prefix.
        if template is None:
            for key in self.REASON_TEMPLATES:
                if feature_name.endswith(key.split("_", 1)[-1]):
                    template = self.REASON_TEMPLATES[key]
                    break

        # Step 3 — fallback
        if template is None:
            return self._FALLBACK_TEMPLATE.format(feature=feature_name)

        # Step 4 — fill {multiplier} placeholder
        if "{multiplier}" in template:
            if feature_value is not None and not np.isnan(float(feature_value)):
                multiplier = round(abs(float(feature_value)), 1)
            else:
                # If feature value is NaN or missing, use the absolute SHAP
                # value as a proxy for "how extreme" this signal was.
                multiplier = round(abs(float(shap_value)), 2)
            template = template.replace("{multiplier}", str(multiplier))

        return template
