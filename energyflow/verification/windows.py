"""Align meter energy and driver values on one frozen window grid.

Energy comes from interval reconstruction: missing readings, meter swaps
without continuous coverage, and long gaps surface as invalid seconds, and
any window past the invalid-second threshold is excluded — never
interpolated into normal production. Driver gaps likewise invalidate a
window instead of being filled.
"""
from fractions import Fraction
from ..values import fraction, rational, fingerprint
from ..calendar import boundaries


class WindowBuilder:
    def __init__(self, reconstructor, drivers):
        self.reconstructor = reconstructor
        self.drivers = drivers

    def driver_value(self, definition, start, end):
        entries = self.drivers.series(definition, start, end)
        covered = [(row["start"], row["end"]) for row in entries]
        cursor = start
        for left, right in sorted(covered):
            if left > cursor:
                return None, entries
            cursor = max(cursor, right)
        if cursor < end:
            return None, entries
        total = Fraction(0)
        mean = entries and entries[0]["definition"] == "mean"
        for row in entries:
            left = max(start, row["start"])
            right = min(end, row["end"])
            if mean:
                total += Fraction(row["value"]) * (right - left)
            else:
                total += Fraction(row["value"]) * Fraction(
                    right - left, row["end"] - row["start"]
                )
        if mean:
            total /= Fraction(end - start)
        return total, entries

    def build(self, payload):
        meter = payload["meter"]
        width = payload["window_seconds"]
        origin = payload["origin"]
        start = payload["baseline_start"]
        end = payload["reporting_end"]
        threshold = payload["thresholds"]["max_invalid_seconds"]
        cuts = boundaries(start, end, width, origin)
        roles = sorted(payload["drivers"])
        windows = []
        for index, (left, right) in enumerate(zip(cuts, cuts[1:])):
            if right <= payload["baseline_end"]:
                period = "baseline"
            elif left >= payload["reporting_start"]:
                period = "reporting"
            else:
                period = "excluded"
            report = self.reconstructor.reconstruct(meter, left, right)
            invalid = report["coverage"]["missing_seconds"]
            measurement_flags = sorted(
                {
                    flag
                    for interval in report["intervals"]
                    if not interval["valid"]
                    for flag in interval["flags"]
                }
            )
            reasons = []
            if invalid > threshold:
                reasons.append("measurement_invalid")
            values = {}
            evidence = {}
            for role in roles:
                value, entries = self.driver_value(payload["drivers"][role], left, right)
                values[role] = value
                evidence[role] = [
                    {"id": row["id"], "revision": row["revision"], "digest": row["digest"]}
                    for row in entries
                ]
                if value is None:
                    reasons.append("driver_gap:" + role)
            valid = not reasons
            windows.append(
                {
                    "index": index,
                    "start": left,
                    "end": right,
                    "period": period,
                    "energy_microkwh": fraction(report["microkwh"]),
                    "invalid_seconds": invalid,
                    "measurement_flags": measurement_flags,
                    "drivers": values,
                    "valid": valid,
                    "reasons": reasons,
                    "input_digest": fingerprint(
                        {
                            "start": left,
                            "end": right,
                            "intervals": [
                                {
                                    "id": row["id"],
                                    "microkwh": row["microkwh"],
                                    "flags": row["flags"],
                                }
                                for row in report["intervals"]
                            ],
                            "drivers": evidence,
                        }
                    ),
                }
            )
        return windows


def public_window(window, roles):
    return {
        "index": window["index"],
        "start": window["start"],
        "end": window["end"],
        "period": window["period"],
        "energy_microkwh": rational(window["energy_microkwh"]),
        "invalid_seconds": window["invalid_seconds"],
        "measurement_flags": window["measurement_flags"],
        "drivers": {
            role: rational(window["drivers"][role])
            if window["drivers"][role] is not None
            else None
            for role in roles
        },
        "valid": window["valid"],
        "reasons": window["reasons"],
        "input_digest": window["input_digest"],
    }
