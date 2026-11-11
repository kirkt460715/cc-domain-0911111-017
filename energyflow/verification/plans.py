"""Verification plans: frozen baseline, retrofit point, and data thresholds.

A plan binds a meter to its driver roles, the baseline and reporting
periods, the retrofit instant, the alignment grid, and every data-quality
threshold. Freezing makes the configuration immutable; evaluations always
reference the frozen digest so later re-computation is comparable.
"""
from ..values import (
    key,
    text,
    instant,
    integer,
    fixed,
    canonical,
    fingerprint,
    reject,
    object_fields,
)

ROLES = ("production", "runtime", "temperature")
ROLE_AGGREGATION = {"production": "sum", "runtime": "sum", "temperature": "mean"}

THRESHOLD_FIELDS = {
    "max_invalid_seconds",
    "min_baseline_windows",
    "min_reporting_windows",
    "correlation_limit",
    "ridge_lambda",
    "validation_folds",
    "block_seconds",
}


class Plans:
    def __init__(self, store, catalog, drivers):
        self.store = store
        self.catalog = catalog
        self.drivers = drivers

    def normalize_thresholds(self, thresholds, window_seconds):
        if thresholds is None:
            thresholds = {}
        object_fields(thresholds, THRESHOLD_FIELDS)
        result = {
            "max_invalid_seconds": integer(
                thresholds.get("max_invalid_seconds", 0), 0, window_seconds - 1
            ),
            "min_baseline_windows": integer(
                thresholds.get("min_baseline_windows", 12), 2, 1000
            ),
            "min_reporting_windows": integer(
                thresholds.get("min_reporting_windows", 3), 1, 1000
            ),
            "correlation_limit": fixed(
                thresholds.get("correlation_limit", "0.9"), signed=True
            ),
            "ridge_lambda": fixed(thresholds.get("ridge_lambda", "0.01")),
            "validation_folds": integer(thresholds.get("validation_folds", 4), 1, 10),
            "block_seconds": integer(
                thresholds.get("block_seconds", 30 * 86400), window_seconds, 366 * 86400
            ),
        }
        if not 0 < result["correlation_limit"] <= 10**6:
            reject("input", "correlation limit must lie within (0, 1]")
        if result["block_seconds"] % window_seconds:
            reject("input", "residual block width must be a whole number of windows")
        return result

    def define(
        self,
        plan,
        meter,
        drivers,
        baseline_start,
        baseline_end,
        reporting_start,
        reporting_end,
        retrofit_at,
        window_seconds=86400,
        origin=0,
        primary=None,
        thresholds=None,
        note="",
    ):
        key(plan)
        self.catalog.get(meter)
        if not isinstance(drivers, dict) or not drivers:
            reject("input", "driver roles must be a nonempty object")
        roles = {}
        for role, ident in drivers.items():
            key(role)
            if role not in ROLES:
                reject("input", "unknown driver role", role=role)
            key(ident)
            if ident in roles.values():
                reject("input", "one driver cannot fill two roles")
            roles[role] = ident
        if "production" not in roles:
            reject("input", "a production driver is required")
        for role, ident in sorted(roles.items()):
            definition = self.drivers.get(ident)
            if definition["meter"] != meter:
                reject("input", "driver belongs to another meter", role=role)
            expected = ROLE_AGGREGATION[role]
            if definition["aggregation"] != expected:
                reject(
                    "input",
                    "driver aggregation does not match its role",
                    role=role,
                    required=expected,
                )
        primary = "production" if primary is None else key(primary)
        if primary not in roles:
            reject("input", "primary driver must be one of the plan roles")
        window_seconds = integer(window_seconds, 300, 31 * 86400)
        origin = instant(origin)
        bounds = {
            "baseline_start": instant(baseline_start),
            "baseline_end": instant(baseline_end),
            "reporting_start": instant(reporting_start),
            "reporting_end": instant(reporting_end),
        }
        for name, moment in bounds.items():
            if (moment - origin) % window_seconds:
                reject("time", "period bound must align to the window grid", bound=name)
        if not bounds["baseline_start"] < bounds["baseline_end"]:
            reject("time", "baseline period must be nonempty")
        if not bounds["reporting_start"] < bounds["reporting_end"]:
            reject("time", "reporting period must be nonempty")
        retrofit_at = instant(retrofit_at)
        if not bounds["baseline_end"] <= retrofit_at <= bounds["reporting_start"]:
            reject("time", "retrofit instant must separate baseline and reporting")
        total_windows = (bounds["reporting_end"] - bounds["baseline_start"]) // (
            window_seconds
        )
        if total_windows > 2000:
            reject("capacity", "plan spans too many windows")
        payload = {
            "id": plan,
            "meter": meter,
            "drivers": {role: roles[role] for role in sorted(roles)},
            "primary": primary,
            "baseline_start": bounds["baseline_start"],
            "baseline_end": bounds["baseline_end"],
            "reporting_start": bounds["reporting_start"],
            "reporting_end": bounds["reporting_end"],
            "retrofit_at": retrofit_at,
            "window_seconds": window_seconds,
            "origin": origin,
            "thresholds": self.normalize_thresholds(thresholds, window_seconds),
            "note": text(note, 1000, True),
        }
        with self.store.atomic() as con:
            old = self.store.one("SELECT id,state FROM plans WHERE id=?", (plan,))
            if old:
                current = self.get(plan)
                if current["payload"] != payload:
                    reject("conflict", "plan identity differs")
                return current
            con.execute(
                "INSERT INTO plans VALUES(?,'draft',1,?,?)",
                (plan, canonical(payload), fingerprint(payload)),
            )
            self.store.event(
                "plan_defined", plan, {"digest": fingerprint(payload)}
            )
        return self.get(plan)

    def get(self, plan):
        row, payload = self.store.payload("plans", plan)
        return {
            "id": plan,
            "state": row["state"],
            "revision": row["revision"],
            "payload": payload,
            "digest": row["digest"],
        }

    def frozen(self, plan):
        record = self.get(plan)
        if record["state"] != "frozen":
            reject("state", "plan must be frozen before evaluation")
        return record

    def freeze(self, plan, expected_digest, operator):
        key(operator)
        with self.store.atomic() as con:
            record = self.get(plan)
            if record["digest"] != expected_digest:
                reject("conflict", "plan review fingerprint differs")
            if record["state"] == "frozen":
                return record
            con.execute(
                "UPDATE plans SET state='frozen',revision=revision+1 WHERE id=?",
                (plan,),
            )
            self.store.event(
                "plan_frozen", plan, {"digest": record["digest"], "operator": operator}
            )
        return self.get(plan)

    def list(self, meter=None):
        result = []
        for row in self.store.rows(
            "SELECT id,state,revision FROM plans ORDER BY id"
        ):
            record = self.get(row["id"])
            if meter is None or record["payload"]["meter"] == meter:
                result.append(
                    {
                        "id": row["id"],
                        "state": row["state"],
                        "revision": row["revision"],
                        "meter": record["payload"]["meter"],
                        "baseline_start": record["payload"]["baseline_start"],
                        "baseline_end": record["payload"]["baseline_end"],
                        "reporting_start": record["payload"]["reporting_start"],
                        "reporting_end": record["payload"]["reporting_end"],
                        "digest": record["digest"],
                    }
                )
        return result
