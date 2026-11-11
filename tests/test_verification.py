import unittest
from fractions import Fraction
from decimal import Decimal
from energyflow.service import EnergyService
from energyflow.values import EnergyError, fraction

DAY = 86400
BASELINE_DAYS = 60
REPORTING_DAYS = 30
SPAN_DAYS = BASELINE_DAYS + REPORTING_DAYS


def production(day):
    if day < BASELINE_DAYS:
        return 60 + (day % 11) * 10
    return 60 + (day % 5) * 6


def runtime(day):
    if day < BASELINE_DAYS:
        return 12 + (day % 4) * 4
    return 12 + (day % 2) * 4


def temperature(day):
    return ((day * 7) % 13) - 6


def model_energy(day):
    return (
        2_000_000
        + 30_000 * production(day)
        + 40_000 * runtime(day)
        + 10_000 * temperature(day)
    )


def actual_energy(day):
    base = model_energy(day)
    if day < BASELINE_DAYS:
        return base
    return 4 * base // 5


class VerificationCase(unittest.TestCase):
    def setUp(self):
        self.s = EnergyService()
        self.addCleanup(self.s.close)
        self.s.catalog.register("m", "Compressor feeder")
        self.s.catalog.install(
            "seg", "m", 0, 0, microkwh_per_tick=1, max_gap=2 * DAY
        )
        cumulative = 0
        for day in range(1, SPAN_DAYS + 1):
            cumulative += actual_energy(day - 1)
            self.s.readings.add(
                dict(id=f"r{day}", segment="seg", at=day * DAY, value=cumulative)
            )
        self.s.drivers.define("prod", "m", "Output units", "sum", "unit")
        self.s.drivers.define("run", "m", "Runtime", "sum", "hour")
        self.s.drivers.define("tmp", "m", "Ambient", "mean", "degC")
        for day in range(SPAN_DAYS):
            self.s.drivers.put(
                f"p{day}", "prod", day * DAY, (day + 1) * DAY, str(production(day))
            )
            self.s.drivers.put(
                f"h{day}", "run", day * DAY, (day + 1) * DAY, str(runtime(day))
            )
            self.s.drivers.put(
                f"t{day}", "tmp", day * DAY, (day + 1) * DAY, str(temperature(day))
            )

    def freeze_plan(self, name="cv", **overrides):
        config = dict(
            meter="m",
            drivers={"production": "prod", "runtime": "run", "temperature": "tmp"},
            baseline_start=0,
            baseline_end=BASELINE_DAYS * DAY,
            reporting_start=BASELINE_DAYS * DAY,
            reporting_end=SPAN_DAYS * DAY,
            retrofit_at=BASELINE_DAYS * DAY,
        )
        config.update(overrides)
        defined = self.s.plans.define(name, **config)
        return self.s.plans.freeze(name, defined["digest"], "engineer")


class SavingsVerificationTests(VerificationCase):
    def test_counterfactual_savings_and_decomposition(self):
        self.freeze_plan()
        report = self.s.verification.evaluate("cv")
        self.assertFalse(report["reused"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["selected_model"], "ridge")
        self.assertEqual(report["baseline"]["valid_windows"], BASELINE_DAYS)
        self.assertEqual(report["reporting"]["valid_windows"], REPORTING_DAYS)
        verified = report["totals"]["verified"]
        self.assertEqual(verified["windows"], REPORTING_DAYS)
        counterfactual = fraction(verified["counterfactual_microkwh"])
        savings = fraction(verified["savings_microkwh"])
        ratio = savings / counterfactual
        self.assertTrue(Fraction(195, 1000) < ratio < Fraction(205, 1000))
        self.assertEqual(
            verified["savings_rounded_microkwh"],
            round(float(savings)),
        )
        split = report["decomposition"]
        condition = fraction(split["operating_condition_change_microkwh"])
        efficiency = fraction(split["efficiency_change_microkwh"])
        self.assertLess(condition, 0)
        self.assertLess(efficiency, 0)
        self.assertEqual(fraction(split["savings_microkwh"]), -efficiency)
        self.assertEqual(
            fraction(split["bill_change_microkwh"]), condition + efficiency
        )
        self.assertEqual(
            fraction(split["bill_change_microkwh"]),
            fraction(verified["actual_microkwh"])
            - REPORTING_DAYS * fraction(split["baseline_mean_microkwh"]),
        )

    def test_forward_chaining_uses_only_earlier_windows(self):
        self.freeze_plan()
        report = self.s.verification.evaluate("cv")
        folds = report["models"]["ridge"]["validation"]["folds"]
        self.assertGreaterEqual(len(folds), 2)
        train_sizes = []
        for fold in folds:
            self.assertLessEqual(fold["train"]["end"], fold["validate"]["start"])
            self.assertEqual(
                fold["standardization"]["window"]["start"], fold["train"]["start"]
            )
            self.assertEqual(
                fold["standardization"]["window"]["end"], fold["train"]["end"]
            )
            train_sizes.append(fold["train"]["windows"])
        self.assertEqual(train_sizes, sorted(train_sizes))
        ridge_error = Decimal(report["models"]["ridge"]["validation"]["cvrmse"])
        simple_error = Decimal(report["models"]["simple"]["validation"]["cvrmse"])
        self.assertLess(ridge_error, simple_error)
        self.assertLess(ridge_error, Decimal("0.01"))

    def test_residual_blocks_flag_significant_savings(self):
        self.freeze_plan()
        report = self.s.verification.evaluate("cv")
        blocks = report["residual_blocks"]
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["windows"], REPORTING_DAYS)
        self.assertFalse(block["within_interval"])
        self.assertEqual(
            block["savings_microkwh"], report["totals"]["verified"]["savings_microkwh"]
        )
        lower = fraction(block["residual_interval"]["lower_microkwh"])
        upper = fraction(block["residual_interval"]["upper_microkwh"])
        self.assertLess(lower, upper)

    def test_evaluation_is_idempotent_and_chained(self):
        self.freeze_plan()
        first = self.s.verification.evaluate("cv")
        again = self.s.verification.evaluate("cv")
        self.assertTrue(again["reused"])
        self.assertEqual(len(self.s.verification.evaluations("cv")), 1)
        self.s.readings.correct(
            "r75",
            1,
            self.s.readings.get("r75")["value"] + 500_000,
            "normal",
            "late meter calibration",
        )
        impact = self.s.verification.impact("cv")
        self.assertTrue(impact["changed"])
        self.assertEqual(
            {row["index"] for row in impact["invalidated_windows"]}, {74, 75}
        )
        second = self.s.verification.evaluate("cv", note="after calibration fix")
        self.assertFalse(second["reused"])
        self.assertEqual(second["seq"], 2)
        self.assertEqual(second["previous"], first["id"] and impact["evaluation_digest"])
        chain = self.s.verification.evaluations("cv")
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[1]["previous"], chain[0]["digest"])
        original = self.s.verification.evaluation("cv", 1)
        self.assertEqual(original["digest"], chain[0]["digest"])
        before_rows = {row["index"]: row for row in original["results"]}
        after_rows = {row["index"]: row for row in second["results"]}
        self.assertNotEqual(
            before_rows[74]["actual_microkwh"], after_rows[74]["actual_microkwh"]
        )
        self.assertNotEqual(
            before_rows[75]["actual_microkwh"], after_rows[75]["actual_microkwh"]
        )
        self.assertEqual(
            before_rows[80]["actual_microkwh"], after_rows[80]["actual_microkwh"]
        )
        self.assertTrue(self.s.store.integrity()["verified"])

    def test_driver_correction_invalidates_window_and_keeps_chain(self):
        self.freeze_plan()
        first = self.s.verification.evaluate("cv")
        entry = self.s.drivers.get_entry("p80")
        self.s.drivers.correct("p80", 1, "55", "late production tally")
        history = self.s.drivers.history("p80")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["value"], entry["value"])
        self.assertEqual(history[1]["reason"], "late production tally")
        impact = self.s.verification.impact("cv")
        self.assertEqual(
            [row["index"] for row in impact["invalidated_windows"]], [80]
        )
        second = self.s.verification.evaluate("cv")
        self.assertEqual(second["seq"], 2)
        self.assertEqual(
            self.s.verification.evaluation("cv", 1)["totals"],
            first["totals"],
        )

    def test_extrapolation_and_zero_production_are_not_verified(self):
        self.s.drivers.correct("p70", 1, "0", "line stopped")
        self.s.drivers.correct("p71", 1, "500", "record output")
        self.freeze_plan()
        report = self.s.verification.evaluate("cv")
        self.assertEqual(report["status"], "verified")
        results = {row["index"]: row for row in report["results"]}
        self.assertEqual(results[70]["category"], "zero_production")
        self.assertIn("zero_production", results[70]["flags"])
        self.assertEqual(results[71]["category"], "extrapolated")
        self.assertIn("extrapolation:production", results[71]["flags"])
        self.assertEqual(report["totals"]["verified"]["windows"], REPORTING_DAYS - 2)
        self.assertEqual(report["totals"]["zero_production"]["windows"], 1)
        self.assertEqual(report["totals"]["extrapolated"]["windows"], 1)

    def test_correlated_features_exclude_multivariate(self):
        for day in range(SPAN_DAYS):
            self.s.drivers.correct(
                f"h{day}",
                1,
                str(2 * production(day)),
                "runtime tracks output exactly",
            )
        self.freeze_plan()
        report = self.s.verification.evaluate("cv")
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["selected_model"], "simple")
        self.assertTrue(report["models"]["ridge"]["excluded"])
        self.assertIn("correlated_features:multivariate_excluded", report["reasons"])
        pairs = report["checks"]["correlated"]
        self.assertEqual(pairs[0]["roles"], ["production", "runtime"])

    def test_insufficient_history_cannot_be_verified(self):
        self.s.plans.define(
            "short",
            "m",
            {"production": "prod", "runtime": "run", "temperature": "tmp"},
            0,
            5 * DAY,
            5 * DAY,
            10 * DAY,
            5 * DAY,
        )
        draft = self.s.plans.get("short")
        with self.assertRaises(EnergyError):
            self.s.verification.evaluate("short")
        self.s.plans.freeze("short", draft["digest"], "engineer")
        report = self.s.verification.evaluate("short")
        self.assertEqual(report["status"], "not_verifiable")
        self.assertIn("insufficient_baseline_windows", report["reasons"])
        self.assertEqual(report["totals"]["verified"]["windows"], 0)
        self.assertEqual(report["residual_blocks"], [])
        self.assertIsNone(report["decomposition"])

    def test_constant_drivers_have_no_usable_model(self):
        for day in range(SPAN_DAYS):
            self.s.drivers.correct(f"p{day}", 1, "105", "flat output")
            self.s.drivers.correct(f"h{day}", 1, "21", "flat runtime")
            self.s.drivers.correct(f"t{day}", 1, "3.5", "flat temperature")
        self.freeze_plan()
        report = self.s.verification.evaluate("cv")
        self.assertEqual(report["status"], "not_verifiable")
        self.assertIn("no_driver_variation", report["reasons"])
        self.assertEqual(
            report["checks"]["constant_roles"],
            ["production", "runtime", "temperature"],
        )

    def test_empty_baseline_reports_insufficient_history(self):
        s = EnergyService()
        self.addCleanup(s.close)
        s.catalog.register("m", "Late feeder")
        s.catalog.install("seg", "m", 0, 0, microkwh_per_tick=1, max_gap=2 * DAY)
        cumulative = 0
        for day in range(BASELINE_DAYS, SPAN_DAYS + 1):
            cumulative += actual_energy(day - 1)
            s.readings.add(
                dict(id=f"r{day}", segment="seg", at=day * DAY, value=cumulative)
            )
        s.drivers.define("prod", "m", "Output", "sum", "unit")
        for day in range(SPAN_DAYS):
            s.drivers.put(f"p{day}", "prod", day * DAY, (day + 1) * DAY, "100")
        s.plans.define(
            "cv",
            "m",
            {"production": "prod"},
            0,
            BASELINE_DAYS * DAY,
            BASELINE_DAYS * DAY,
            SPAN_DAYS * DAY,
            BASELINE_DAYS * DAY,
        )
        s.plans.freeze("cv", s.plans.get("cv")["digest"], "engineer")
        report = s.verification.evaluate("cv")
        self.assertEqual(report["status"], "not_verifiable")
        self.assertIn("insufficient_baseline_windows", report["reasons"])
        self.assertEqual(report["baseline"]["valid_windows"], 0)

    def test_plan_freeze_and_immutability(self):
        defined = self.s.plans.define(
            "cv",
            "m",
            {"production": "prod", "runtime": "run", "temperature": "tmp"},
            0,
            BASELINE_DAYS * DAY,
            BASELINE_DAYS * DAY,
            SPAN_DAYS * DAY,
            BASELINE_DAYS * DAY,
        )
        same = self.s.plans.define(
            "cv",
            "m",
            {"production": "prod", "runtime": "run", "temperature": "tmp"},
            0,
            BASELINE_DAYS * DAY,
            BASELINE_DAYS * DAY,
            SPAN_DAYS * DAY,
            BASELINE_DAYS * DAY,
        )
        self.assertEqual(defined["digest"], same["digest"])
        with self.assertRaises(EnergyError):
            self.s.plans.define(
                "cv",
                "m",
                {"production": "prod", "runtime": "run", "temperature": "tmp"},
                0,
                BASELINE_DAYS * DAY,
                BASELINE_DAYS * DAY,
                SPAN_DAYS * DAY,
                BASELINE_DAYS * DAY + 3600,
            )
        with self.assertRaises(EnergyError):
            self.s.plans.freeze("cv", "wrong", "engineer")
        with self.assertRaises(EnergyError):
            self.s.plans.define(
                "bad",
                "m",
                {"runtime": "run"},
                0,
                BASELINE_DAYS * DAY,
                BASELINE_DAYS * DAY,
                SPAN_DAYS * DAY,
                BASELINE_DAYS * DAY,
            )
        with self.assertRaises(EnergyError):
            self.s.plans.define(
                "bad2",
                "m",
                {"production": "prod"},
                0,
                BASELINE_DAYS * DAY,
                BASELINE_DAYS * DAY,
                SPAN_DAYS * DAY,
                BASELINE_DAYS * DAY,
                thresholds={"unknown": 1},
            )
        frozen = self.freeze_plan()
        again = self.s.plans.freeze("cv", frozen["digest"], "engineer")
        self.assertEqual(again["state"], "frozen")
        self.assertEqual(
            [row["id"] for row in self.s.plans.list("m")], ["cv"]
        )


class DataIntegrityTests(VerificationCase):
    def test_long_gap_windows_are_excluded_not_interpolated(self):
        s = EnergyService()
        self.addCleanup(s.close)
        s.catalog.register("m", "Compressor feeder")
        s.catalog.install("seg", "m", 0, 0, microkwh_per_tick=1, max_gap=2 * DAY)
        cumulative = 0
        for day in range(1, SPAN_DAYS + 1):
            cumulative += actual_energy(day - 1)
            if 70 <= day <= 73:
                continue
            s.readings.add(dict(id=f"r{day}", segment="seg", at=day * DAY, value=cumulative))
        for role, ident, values in [
            ("prod", "p", production),
            ("run", "h", runtime),
            ("tmp", "t", temperature),
        ]:
            s.drivers.define(role, "m", role, "sum" if role != "tmp" else "mean", "u")
            for day in range(SPAN_DAYS):
                s.drivers.put(
                    f"{ident}{day}", role, day * DAY, (day + 1) * DAY, str(values(day))
                )
        s.plans.define(
            "cv",
            "m",
            {"production": "prod", "runtime": "run", "temperature": "tmp"},
            0,
            BASELINE_DAYS * DAY,
            BASELINE_DAYS * DAY,
            SPAN_DAYS * DAY,
            BASELINE_DAYS * DAY,
        )
        s.plans.freeze("cv", s.plans.get("cv")["digest"], "engineer")
        table = s.verification.windows("cv")
        invalid = {
            row["index"]: row for row in table["windows"] if not row["valid"]
        }
        self.assertEqual(set(invalid), {69, 70, 71, 72, 73})
        for row in invalid.values():
            self.assertIn("measurement_invalid", row["reasons"])
            self.assertIn("gap_too_long", row["measurement_flags"])
        report = s.verification.evaluate("cv")
        covered = {row["index"] for row in report["results"]}
        self.assertTrue(covered.isdisjoint({69, 70, 71, 72, 73}))
        self.assertEqual(report["totals"]["verified"]["windows"], REPORTING_DAYS - 5)
        self.assertEqual(report["reporting"]["invalid_windows"], 5)

    def test_missing_reading_marks_windows_invalid(self):
        self.s.readings.correct("r75", 1, 0, "missing", "meter unavailable")
        self.freeze_plan()
        table = self.s.verification.windows("cv")
        invalid = {
            row["index"]
            for row in table["windows"]
            if not row["valid"] and row["period"] == "reporting"
        }
        self.assertEqual(invalid, {74, 75})
        report = self.s.verification.evaluate("cv")
        self.assertEqual(report["totals"]["verified"]["windows"], REPORTING_DAYS - 2)

    def test_meter_swap_with_and_without_gap(self):
        for with_gap in (False, True):
            s = EnergyService()
            self.addCleanup(s.close)
            s.catalog.register("m", "Compressor feeder")
            s.catalog.install("seg1", "m", 0, 0, microkwh_per_tick=1, max_gap=2 * DAY)
            cumulative = 0
            for day in range(1, 31):
                cumulative += actual_energy(day - 1)
                s.readings.add(
                    dict(id=f"a{day}", segment="seg1", at=day * DAY, value=cumulative)
                )
            s.catalog.replacement("seg1", "seg2", 30 * DAY, 0, microkwh_per_tick=1)
            cumulative = 0
            for day in range(31, SPAN_DAYS + 1):
                cumulative += actual_energy(day - 1)
                if with_gap and day <= 33:
                    continue
                s.readings.add(
                    dict(id=f"b{day}", segment="seg2", at=day * DAY, value=cumulative)
                )
            for role, ident, values, aggregation in [
                ("prod", "p", production, "sum"),
                ("run", "h", runtime, "sum"),
                ("tmp", "t", temperature, "mean"),
            ]:
                s.drivers.define(role, "m", role, aggregation, "u")
                for day in range(SPAN_DAYS):
                    s.drivers.put(
                        f"{ident}{day}",
                        role,
                        day * DAY,
                        (day + 1) * DAY,
                        str(values(day)),
                    )
            s.plans.define(
                "cv",
                "m",
                {"production": "prod", "runtime": "run", "temperature": "tmp"},
                0,
                BASELINE_DAYS * DAY,
                BASELINE_DAYS * DAY,
                SPAN_DAYS * DAY,
                BASELINE_DAYS * DAY,
            )
            s.plans.freeze("cv", s.plans.get("cv")["digest"], "engineer")
            report = s.verification.evaluate("cv")
            if with_gap:
                self.assertEqual(report["reporting"]["invalid_windows"], 0)
                self.assertEqual(report["baseline"]["invalid_windows"], 4)
                self.assertEqual(
                    report["baseline"]["valid_windows"], BASELINE_DAYS - 4
                )
            else:
                self.assertEqual(report["status"], "verified")
                self.assertEqual(report["baseline"]["valid_windows"], BASELINE_DAYS)

    def test_driver_gap_invalidates_window(self):
        s = EnergyService()
        self.addCleanup(s.close)
        s.catalog.register("m", "Compressor feeder")
        s.catalog.install("seg", "m", 0, 0, microkwh_per_tick=1, max_gap=2 * DAY)
        cumulative = 0
        for day in range(1, SPAN_DAYS + 1):
            cumulative += actual_energy(day - 1)
            s.readings.add(dict(id=f"r{day}", segment="seg", at=day * DAY, value=cumulative))
        s.drivers.define("prod", "m", "Output", "sum", "unit")
        for day in range(SPAN_DAYS):
            if day == 80:
                continue
            s.drivers.put(f"p{day}", "prod", day * DAY, (day + 1) * DAY, "100")
        s.plans.define(
            "cv",
            "m",
            {"production": "prod"},
            0,
            BASELINE_DAYS * DAY,
            BASELINE_DAYS * DAY,
            SPAN_DAYS * DAY,
            BASELINE_DAYS * DAY,
        )
        s.plans.freeze("cv", s.plans.get("cv")["digest"], "engineer")
        table = s.verification.windows("cv")
        row = table["windows"][80]
        self.assertFalse(row["valid"])
        self.assertIn("driver_gap:production", row["reasons"])
        report = s.verification.evaluate("cv")
        self.assertNotIn(80, {r["index"] for r in report["results"]})

    def test_settlement_chain_untouched_by_verification(self):
        self.s.production.put("cal", "m", 0, SPAN_DAYS * DAY, {"a": 1})
        self.s.tariffs.put("rate", "m", 0, SPAN_DAYS * DAY, "1")
        preview = self.s.closing.preview(["m"], 0, 30 * DAY)
        frozen = self.s.closing.freeze(
            "settlement", ["m"], 0, 30 * DAY, preview["digest"], "controller"
        )
        self.freeze_plan()
        self.s.verification.evaluate("cv")
        self.s.readings.correct(
            "r75",
            1,
            self.s.readings.get("r75")["value"] + 1000,
            "normal",
            "calibration",
        )
        self.s.drivers.correct("p75", 1, "61", "restated output")
        self.s.verification.evaluate("cv")
        self.assertEqual(self.s.closing.get("settlement")["digest"], frozen["digest"])
        self.assertTrue(self.s.closing.verify("settlement")["verified"])
        self.assertEqual(len(self.s.verification.evaluations("cv")), 2)
        self.assertTrue(self.s.quality.system()["ok"])


if __name__ == "__main__":
    unittest.main()
