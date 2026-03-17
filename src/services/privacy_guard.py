# src/services/privacy_guard.py

import hashlib
import re
from datetime import datetime


class PrivacyGuard:

    ALLOWED_LOG_FIELDS = {"transaction_id", "decision", "risk_score", "timestamp"}

    # ── Public methods ───────────────────────────────────────────

    def hash_user_id(self, user_id: str) -> str:
        return hashlib.sha256(user_id.encode()).hexdigest()

    def mask_ip(self, ip_address: str) -> str:
        parts = ip_address.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.XXX"
        return "XXX.XXX.XXX.XXX"

    def mask_device_id(self, device_id: str) -> str:
        if len(device_id) <= 4:
            return "****"
        return device_id[:4] + "*" * (len(device_id) - 4)

    def sanitize_for_log(self, result: dict) -> dict:
        """Strip everything except ALLOWED_LOG_FIELDS before writing to logs."""
        return {k: v for k, v in result.items() if k in self.ALLOWED_LOG_FIELDS}

    def validate_input_fields(self, transaction: dict) -> bool:
        """
        Reject the transaction if any field looks like a raw card number.
        Card numbers are 13–19 consecutive digits (covers Visa, MC, Amex, UnionPay).
        """
        for field, value in transaction.items():
            if self._contains_card_number(str(value)):
                raise ValueError(
                    f"Field '{field}' appears to contain a raw card number. "
                    f"PAN must never be passed to the scoring API."
                )
        return True

    # ── Private ──────────────────────────────────────────────────

    def _contains_card_number(self, value: str) -> bool:
        return bool(re.search(r'\b\d{13,19}\b', value))
