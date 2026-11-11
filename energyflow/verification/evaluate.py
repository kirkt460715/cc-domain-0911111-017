"""Savings evaluation: counterfactual baseline, verification, and chains.

The simple baseline is univariate least squares on the primary driver; the
multivariate baseline is ridge regression on standardized drivers. Forward
validation and standardization statistics use only the training rows of each
fold. Savings are reported as verified only where windows are valid, inside
the training range, and producing; extrapolated, zero-production, correlated
or under-sampled cases are exposed instead of being dressed up as savings.
"""
from fractions import Fraction
from ..values import (
    text,
    fraction,
    rational,
    round_even,
    fingerprint,
    canonical,
    reject,
    EnergyError,
)
from . import model
from .windows import WindowBuilder, public_window


def ratio_text(value):
    if value is None:
        return None
    from decimal import Decimal

    return format(
        Decimal(value.numerator) / Decimal(value.denominator), ".6f"
    )


def rounded(value):
    return round_even(Fraction(value))


class Verification:
    def __init__(self, store, catalog, reconstructor, drivers, plans):
        self.store = store
        self.catalog = catalog
        self.reconstructor = reconstructor
        self.drivers = drivers
        self.plans = plans
        self.builder = WindowBuilder(reconstructor, drivers)

    def windows(self, plan):
        record = self.plans.frozen(plan)
        payload = record["payload"]
        roles = sorted(payload["drivers"])
        built = self.builder.build(payload)
        counts = {}
        for window in built:
            key = (window["period"], window["valid"])
            counts[key] = counts.get(key, 0) + 1
        return {
            "plan": plan,
            "plan_digest": record["digest"],
            "windows": [public_window(window, roles) for window in built],
            "counts": [
                {"period": period, "valid": valid, "windows": counts[(period, valid)]}
                for (period, valid) in sorted(counts)
            ],
        }

    def fit_simple(self, rows, primary):
        intercept, slope = model.ols_univariate(
            [row["drivers"][primary] for row in rows],
            [row["energy_microkwh"] for row in rows],
        )
        intercept = intercept.limit_denominator(10**9)
        slope = slope.limit_denominator(10**9)
        degenerate = len({row["drivers"][primary] for row in rows}) <= 1

        def predict(window):
            return intercept + slope * window["drivers"][primary]

        return {
            "kind": "simple",
            "driver": primary,
            "intercept_microkwh": intercept,
            "slope": slope,
            "degenerate": degenerate,
            "predict": predict,
        }

    def fit_ridge(self, rows, roles, penalty):
        columns = [[row["drivers"][role] for row in rows] for role in roles]
        target = [row["energy_microkwh"] for row in rows]
        means, scales, scaled = model.standardize(columns)
        target_mean = model.mean(target)
        target_scale = model.fraction_sqrt(model.variance(target))
        if target_scale == 0:
            reject("model", "baseline energy is constant; no baseline to fit")
        scaled_target = [(value - target_mean) / target_scale for value in target]
        coefficients = [
            beta.limit_denominator(10**9)
            for beta in model.ridge_fit(scaled, scaled_target, penalty)
        ]

        def predict(window):
            total = Fraction(0)
            for role, center, spread, beta in zip(roles, means, scales, coefficients):
                if spread is None:
                    continue
                total += beta * (window["drivers"][role] - center) / spread
            return target_mean + target_scale * total

        return {
            "kind": "ridge",
            "roles": roles,
            "penalty": penalty,
            "means": means,
            "scales": scales,
            "target_mean_microkwh": target_mean,
            "target_scale": target_scale,
            "coefficients": coefficients,
            "constant_roles": [
                role for role, spread in zip(roles, scales) if spread is None
            ],
            "predict": predict,
        }

    def forward_validate(self, rows, fit, folds):
        chains = model.forward_chains(len(rows), folds)
        actual = []
        predicted = []
        detail = []
        for fold, (train_rows, valid_rows) in enumerate(chains, 1):
            train = [rows[index] for index in train_rows]
            validate = [rows[index] for index in valid_rows]
            fitted = fit(train)
            for window in validate:
                actual.append(window["energy_microkwh"])
                predicted.append(round_even(fitted["predict"](window)))
            entry = {
                "fold": fold,
                "train": {
                    "windows": len(train),
                    "start": train[0]["start"],
                    "end": train[-1]["end"],
                },
                "validate": {
                    "windows": len(validate),
                    "start": validate[0]["start"],
                    "end": validate[-1]["end"],
                },
            }
            if fitted["kind"] == "ridge":
                entry["standardization"] = {
                    "means": [rational(value) for value in fitted["means"]],
                    "scales": [
                        rational(value) if value is not None else None
                        for value in fitted["scales"]
                    ],
                    "window": {
                        "start": train[0]["start"],
                        "end": train[-1]["end"],
                    },
                }
            detail.append(entry)
        result = model.metrics(actual, predicted)
        result["folds"] = detail
        return result

    def assess(self, payload, baseline, reporting):
        roles = sorted(payload["drivers"])
        primary = payload["primary"]
        thresholds = payload["thresholds"]
        checks = []
        blocking = []
        if len(baseline) < thresholds["min_baseline_windows"]:
            blocking.append("insufficient_baseline_windows")
        if len(reporting) < thresholds["min_reporting_windows"]:
            blocking.append("insufficient_reporting_windows")
        constant = []
        correlated = []
        if len(baseline) >= 2:
            for role in roles:
                values = {row["drivers"][role] for row in baseline}
                if len(values) <= 1:
                    constant.append(role)
            if len(constant) == len(roles):
                blocking.append("no_driver_variation")
            limit = Fraction(thresholds["correlation_limit"], 10**6)
            for left_index, left in enumerate(roles):
                for right in roles[left_index + 1 :]:
                    value = model.pearson(
                        [row["drivers"][left] for row in baseline],
                        [row["drivers"][right] for row in baseline],
                    )
                    if abs(value) >= limit:
                        correlated.append(
                            {"roles": [left, right], "correlation": ratio_text(value)}
                        )
        if correlated:
            checks.append("correlated_features")
        if primary in constant:
            checks.append("primary_driver_constant")
        return {
            "checks": checks,
            "blocking": blocking,
            "constant_roles": constant,
            "correlated": correlated,
        }

    def report(self, record, built, snapshot, note):
        payload = record["payload"]
        roles = sorted(payload["drivers"])
        primary = payload["primary"]
        thresholds = payload["thresholds"]
        baseline = [
            row for row in built if row["period"] == "baseline" and row["valid"]
        ]
        reporting = [
            row for row in built if row["period"] == "reporting" and row["valid"]
        ]
        assessment = self.assess(payload, baseline, reporting)
        models = {}
        folds = thresholds["validation_folds"]
        if len(baseline) >= 2:
            simple_fit = self.fit_simple(baseline, primary)
            simple_validation = self.forward_validate(
                baseline, lambda rows: self.fit_simple(rows, primary), folds
            )
            models["simple"] = {
                "driver": primary,
                "intercept_microkwh": rational(simple_fit["intercept_microkwh"]),
                "slope": rational(simple_fit["slope"]),
                "degenerate": simple_fit["degenerate"],
                "validation": self.public_metrics(simple_validation),
            }
            ridge_error = None
            try:
                ridge_fit = self.fit_ridge(
                    baseline, roles, Fraction(thresholds["ridge_lambda"], 10**6)
                )
                ridge_validation = self.forward_validate(
                    baseline,
                    lambda rows: self.fit_ridge(
                        rows, roles, Fraction(thresholds["ridge_lambda"], 10**6)
                    ),
                    folds,
                )
            except EnergyError as exc:
                ridge_fit = None
                ridge_error = exc.document()
            models["ridge"] = {
                "roles": roles,
                "penalty": rational(Fraction(thresholds["ridge_lambda"], 10**6)),
                "excluded": bool(assessment["correlated"]),
                "error": ridge_error,
            }
            if ridge_fit is not None:
                models["ridge"].update(
                    {
                        "coefficients": {
                            role: rational(beta)
                            for role, beta in zip(roles, ridge_fit["coefficients"])
                        },
                        "constant_roles": ridge_fit["constant_roles"],
                        "validation": self.public_metrics(ridge_validation),
                    }
                )
        else:
            simple_fit = None
            ridge_fit = None
        selected = None
        selection_note = None
        if simple_fit is not None:
            candidates = ["simple"]
            if (
                ridge_fit is not None
                and not assessment["correlated"]
                and not models["ridge"]["excluded"]
            ):
                candidates.append("ridge")
            if len(candidates) == 2:
                simple_error = fraction(models["simple"]["validation"]["rmse_microkwh"])
                ridge_error_value = fraction(
                    models["ridge"]["validation"]["rmse_microkwh"]
                )
                selected = "ridge" if ridge_error_value < simple_error else "simple"
                selection_note = "lower_validation_error"
            else:
                selected = "simple"
                if assessment["correlated"]:
                    selection_note = "ridge_excluded_correlated_features"
                elif ridge_fit is None:
                    selection_note = "ridge_unavailable"
            if models["simple"]["degenerate"] and selected == "simple":
                if ridge_fit is not None and "ridge" in candidates:
                    selected = "ridge"
                    selection_note = "simple_degenerate"
                else:
                    assessment["blocking"].append("no_usable_model")
        usable = not assessment["blocking"]
        final_fit = None
        if usable:
            final_fit = (
                self.fit_simple(baseline, primary)
                if selected == "simple"
                else self.fit_ridge(
                    baseline, roles, Fraction(thresholds["ridge_lambda"], 10**6)
                )
            )
        ranges = {
            role: (
                min(row["drivers"][role] for row in baseline),
                max(row["drivers"][role] for row in baseline),
            )
            for role in roles
        } if baseline else {}
        results = []
        if final_fit is not None:
            for window in reporting:
                flags = []
                for role in roles:
                    low, high = ranges[role]
                    value = window["drivers"][role]
                    if value < low or value > high:
                        flags.append("extrapolation:" + role)
                if window["drivers"]["production"] == 0:
                    flags.append("zero_production")
                counterfactual = Fraction(round_even(final_fit["predict"](window)))
                actual = window["energy_microkwh"]
                category = "verified"
                if "zero_production" in flags:
                    category = "zero_production"
                elif any(flag.startswith("extrapolation") for flag in flags):
                    category = "extrapolated"
                results.append(
                    {
                        "index": window["index"],
                        "start": window["start"],
                        "end": window["end"],
                        "actual_microkwh": actual,
                        "counterfactual_microkwh": counterfactual,
                        "savings_microkwh": counterfactual - actual,
                        "flags": flags,
                        "category": category,
                    }
                )
        totals = self.totals(results)
        decomposition = None
        if final_fit is not None and results:
            baseline_mean = model.mean(
                [
                    Fraction(round_even(final_fit["predict"](row)))
                    for row in baseline
                ]
            )
            decomposition = self.decompose(
                [row for row in results if row["category"] == "verified"],
                baseline_mean,
            )
        blocks = []
        if final_fit is not None and results:
            sigma = fraction(models[selected]["validation"]["rmse_microkwh"])
            blocks = self.residual_blocks(
                payload, [row for row in results if row["category"] == "verified"], sigma
            )
        status = "verified" if usable else "not_verifiable"
        reasons = list(assessment["blocking"])
        if assessment["correlated"]:
            reasons.append("correlated_features:multivariate_excluded")
        if models.get("simple", {}).get("degenerate"):
            reasons.append("primary_driver_constant")
        report = {
            "plan": record["id"],
            "plan_digest": record["digest"],
            "snapshot": snapshot,
            "note": text(note, 1000, True),
            "status": status,
            "reasons": reasons,
            "selected_model": selected,
            "selection_note": selection_note,
            "windows": [public_window(row, roles) for row in built],
            "baseline": {
                "valid_windows": len(baseline),
                "invalid_windows": sum(
                    row["period"] == "baseline" and not row["valid"] for row in built
                ),
                "driver_ranges": {
                    role: {"min": rational(low), "max": rational(high)}
                    for role, (low, high) in sorted(ranges.items())
                },
            },
            "reporting": {
                "valid_windows": len(reporting),
                "invalid_windows": sum(
                    row["period"] == "reporting" and not row["valid"] for row in built
                ),
            },
            "checks": {
                "constant_roles": assessment["constant_roles"],
                "correlated": assessment["correlated"],
                "findings": assessment["checks"],
            },
            "models": models,
            "results": [self.public_result(row) for row in results],
            "totals": totals,
            "decomposition": decomposition,
            "residual_blocks": blocks,
            "format_version": 1,
        }
        return report

    def public_metrics(self, result):
        return {
            "rmse_microkwh": rational(result["rmse"]),
            "cvrmse": ratio_text(result["cvrmse"]),
            "nmbe": ratio_text(result["nmbe"]),
            "mean_actual_microkwh": rational(result["mean_actual"]),
            "folds": result["folds"],
        }

    def public_result(self, row):
        return {
            "index": row["index"],
            "start": row["start"],
            "end": row["end"],
            "actual_microkwh": rational(row["actual_microkwh"]),
            "counterfactual_microkwh": rational(row["counterfactual_microkwh"]),
            "savings_microkwh": rational(row["savings_microkwh"]),
            "flags": row["flags"],
            "category": row["category"],
        }

    def totals(self, results):
        groups = {}
        for row in results:
            bucket = groups.setdefault(
                row["category"],
                {"windows": 0, "actual": Fraction(0), "counterfactual": Fraction(0)},
            )
            bucket["windows"] += 1
            bucket["actual"] += row["actual_microkwh"]
            bucket["counterfactual"] += row["counterfactual_microkwh"]
        output = {}
        for category in ["verified", "extrapolated", "zero_production"]:
            bucket = groups.get(
                category, {"windows": 0, "actual": Fraction(0), "counterfactual": Fraction(0)}
            )
            savings = bucket["counterfactual"] - bucket["actual"]
            output[category] = {
                "windows": bucket["windows"],
                "actual_microkwh": rational(bucket["actual"]),
                "counterfactual_microkwh": rational(bucket["counterfactual"]),
                "savings_microkwh": rational(savings),
                "savings_rounded_microkwh": rounded(savings),
            }
        return output

    def decompose(self, verified, baseline_mean):
        count = len(verified)
        actual = sum((row["actual_microkwh"] for row in verified), Fraction(0))
        counterfactual = sum(
            (row["counterfactual_microkwh"] for row in verified), Fraction(0)
        )
        condition_change = counterfactual - baseline_mean * count
        efficiency_change = actual - counterfactual
        return {
            "windows": count,
            "baseline_mean_microkwh": rational(baseline_mean),
            "bill_change_microkwh": rational(actual - baseline_mean * count),
            "operating_condition_change_microkwh": rational(condition_change),
            "efficiency_change_microkwh": rational(efficiency_change),
            "savings_microkwh": rational(-efficiency_change),
            "sign_convention": "change = reporting - baseline; savings = -efficiency_change",
        }

    def residual_blocks(self, payload, verified, sigma):
        width = payload["thresholds"]["block_seconds"]
        origin = payload["origin"]
        groups = {}
        for row in verified:
            block = (row["start"] - origin) // width
            groups.setdefault(block, []).append(row)
        blocks = []
        for block in sorted(groups):
            rows = groups[block]
            count = len(rows)
            actual = sum((row["actual_microkwh"] for row in rows), Fraction(0))
            counterfactual = sum(
                (row["counterfactual_microkwh"] for row in rows), Fraction(0)
            )
            band = (2 * sigma * model.fraction_sqrt(count)).limit_denominator(10**6)
            lower = counterfactual - band
            upper = counterfactual + band
            blocks.append(
                {
                    "start": origin + block * width,
                    "end": origin + (block + 1) * width,
                    "windows": count,
                    "actual_microkwh": rational(actual),
                    "counterfactual_microkwh": rational(counterfactual),
                    "savings_microkwh": rational(counterfactual - actual),
                    "residual_interval": {
                        "lower_microkwh": rational(lower),
                        "upper_microkwh": rational(upper),
                    },
                    "within_interval": lower <= actual <= upper,
                }
            )
        return blocks

    def latest(self, plan):
        row = self.store.one(
            "SELECT id,seq,digest FROM evaluations WHERE plan=? ORDER BY seq DESC LIMIT 1",
            (plan,),
        )
        if row is None:
            return None
        _, payload = self.store.payload("evaluations", row["id"])
        return {"id": row["id"], "seq": row["seq"], "digest": row["digest"], "payload": payload}

    def evaluate(self, plan, note=""):
        record = self.plans.frozen(plan)
        built = self.builder.build(record["payload"])
        snapshot = fingerprint(
            {"plan": record["digest"], "windows": [row["input_digest"] for row in built]}
        )
        with self.store.atomic() as con:
            last = self.latest(plan)
            if last and last["payload"]["snapshot"] == snapshot:
                return {**last["payload"], "id": last["id"], "reused": True}
            report = self.report(record, built, snapshot, note)
            seq = last["seq"] + 1 if last else 1
            ident = plan + ".e" + str(seq)
            previous = last["digest"] if last else "0" * 64
            stored = {
                "id": ident,
                "plan": plan,
                "seq": seq,
                "previous": previous,
                **report,
            }
            digest = fingerprint(stored)
            con.execute(
                "INSERT INTO evaluations VALUES(?,?,?,?,?,?)",
                (ident, plan, seq, canonical(stored), digest, previous),
            )
            self.store.event(
                "verification_evaluated",
                ident,
                {"plan": plan, "seq": seq, "digest": digest, "status": report["status"]},
            )
        return {**stored, "reused": False}

    def evaluations(self, plan):
        self.plans.get(plan)
        result = []
        for row in self.store.rows(
            "SELECT id,seq,digest FROM evaluations WHERE plan=? ORDER BY seq", (plan,)
        ):
            _, payload = self.store.payload("evaluations", row["id"])
            result.append(
                {
                    "id": row["id"],
                    "seq": row["seq"],
                    "digest": row["digest"],
                    "previous": payload["previous"],
                    "status": payload["status"],
                    "verified_savings_microkwh": payload["totals"]["verified"][
                        "savings_microkwh"
                    ],
                }
            )
        return result

    def evaluation(self, plan, seq):
        from ..values import integer

        integer(seq, 1)
        row = self.store.one(
            "SELECT id FROM evaluations WHERE plan=? AND seq=?", (plan, seq)
        )
        if row is None:
            reject("missing", "evaluation not found")
        _, payload = self.store.payload("evaluations", row["id"])
        return {**payload, "id": row["id"], "digest": fingerprint(payload)}

    def impact(self, plan):
        record = self.plans.frozen(plan)
        last = self.latest(plan)
        if last is None:
            reject("missing", "no evaluation exists for impact analysis")
        built = self.builder.build(record["payload"])
        before = {row["index"]: row for row in last["payload"]["windows"]}
        changed = []
        for window in built:
            old = before.get(window["index"])
            if old is None:
                continue
            if old["input_digest"] != window["input_digest"] or old["valid"] != window["valid"]:
                changed.append(
                    {
                        "index": window["index"],
                        "start": window["start"],
                        "end": window["end"],
                        "period": window["period"],
                        "before": {
                            "valid": old["valid"],
                            "reasons": old["reasons"],
                            "input_digest": old["input_digest"],
                        },
                        "after": {
                            "valid": window["valid"],
                            "reasons": window["reasons"],
                            "input_digest": window["input_digest"],
                        },
                    }
                )
        snapshot = fingerprint(
            {"plan": record["digest"], "windows": [row["input_digest"] for row in built]}
        )
        return {
            "plan": plan,
            "evaluation": last["id"],
            "evaluation_digest": last["digest"],
            "changed": snapshot != last["payload"]["snapshot"],
            "invalidated_windows": changed,
            "invalidated_count": len(changed),
        }
