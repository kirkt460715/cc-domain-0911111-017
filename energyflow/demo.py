from tempfile import TemporaryDirectory
from pathlib import Path
from .service import EnergyService


def verification(service):
    width = 86400
    service.catalog.register("line2", "Synthetic compressor feeder")
    service.catalog.install(
        "feed", "line2", 0, 0, microkwh_per_tick=1, max_gap=2 * width
    )
    service.drivers.define("units", "line2", "Output", "sum", "unit")
    service.drivers.define("hours", "line2", "Runtime", "sum", "hour")
    service.drivers.define("ambient", "line2", "Temperature", "mean", "degC")
    baseline_days = 30
    reporting_days = 15

    def output(day):
        return 60 + (day % 11) * 10 if day < baseline_days else 60 + (day % 5) * 6

    def runtime(day):
        return 12 + (day % 4) * 4 if day < baseline_days else 12 + (day % 2) * 4

    def energy(day):
        model = 2_000_000 + 30_000 * output(day) + 40_000 * runtime(day)
        return model if day < baseline_days else 4 * model // 5

    cumulative = 0
    for day in range(1, baseline_days + reporting_days + 1):
        cumulative += energy(day - 1)
        service.readings.add(
            dict(id=f"feed-{day}", segment="feed", at=day * width, value=cumulative)
        )
    for day in range(baseline_days + reporting_days):
        service.drivers.put(
            f"u{day}", "units", day * width, (day + 1) * width, str(output(day))
        )
        service.drivers.put(
            f"h{day}", "hours", day * width, (day + 1) * width, str(runtime(day))
        )
        service.drivers.put(f"t{day}", "ambient", day * width, (day + 1) * width, "18")
    retrofit = baseline_days * width
    plan = service.plans.define(
        "compressor",
        "line2",
        {"production": "units", "runtime": "hours", "temperature": "ambient"},
        0,
        retrofit,
        retrofit,
        (baseline_days + reporting_days) * width,
        retrofit,
        thresholds={"min_baseline_windows": 12, "min_reporting_windows": 3},
    )
    service.plans.freeze("compressor", plan["digest"], "verifier")
    report = service.verification.evaluate("compressor")
    assert report["status"] == "verified"
    verified = report["totals"]["verified"]
    assert verified["windows"] == reporting_days
    from .values import fraction

    savings = fraction(verified["savings_microkwh"])
    counterfactual = fraction(verified["counterfactual_microkwh"])
    ratio = savings / counterfactual
    assert ratio > fraction("0.19") and ratio < fraction("0.21")
    split = report["decomposition"]
    assert fraction(split["operating_condition_change_microkwh"]) < 0
    assert fraction(split["efficiency_change_microkwh"]) < 0
    assert not service.verification.impact("compressor")["changed"]
    service.readings.correct(
        "feed-40",
        1,
        service.readings.get("feed-40")["value"] + 1000,
        "normal",
        "synthetic late calibration",
    )
    impact = service.verification.impact("compressor")
    assert impact["changed"] and impact["invalidated_count"] == 2
    followup = service.verification.evaluate("compressor")
    assert followup["seq"] == 2 and followup["previous"] == impact["evaluation_digest"]
    assert len(service.verification.evaluations("compressor")) == 2
    return {
        "status": report["status"],
        "selected_model": report["selected_model"],
        "verified_windows": verified["windows"],
        "savings_microkwh": verified["savings_rounded_microkwh"],
        "evaluations": 2,
    }


def run():
    with TemporaryDirectory() as folder:
        with EnergyService(Path(folder) / "energy.db") as service:
            service.catalog.register("line", "Synthetic production line")
            service.catalog.install("counter", "line", 0, 0, max_gap=7200)
            service.readings.import_batch(
                "shift",
                [
                    {"id": "r1", "segment": "counter", "at": 3600, "value": 10},
                    {"id": "r2", "segment": "counter", "at": 7200, "value": 30},
                ],
            )
            service.production.put(
                "first", "line", 0, 3600, {"productA": 1, "productB": 1}
            )
            service.production.put("second", "line", 3600, 7200, {"productB": 1})
            service.tariffs.put("offpeak", "line", 0, 3600, "0.5")
            service.tariffs.put("peak", "line", 3600, 7200, "1.0")
            service.emissions.put("local", "line", 0, 7200, "500", "synthetic factor")
            preview = service.closing.preview(["line"], 0, 7200, require_factors=True)
            frozen = service.closing.freeze(
                "shift",
                ["line"],
                0,
                7200,
                preview["digest"],
                "controller",
                require_factors=True,
            )
            assert (
                sum(row["microkwh"] for row in frozen["payload"]["energy"]) == 30000000
            )
            assert sum(row["minor"] for row in frozen["payload"]["costs"]) == 2500
            service.readings.correct(
                "r1", 1, 12, "normal", "synthetic calibration correction"
            )
            correction = service.closing.propose_correction(
                "adjust", "shift", "corrected intermediate counter"
            )
            service.closing.approve_correction(
                "adjust", correction["digest"], "controller"
            )
            effective = service.closing.effective("shift")
            assert (
                sum(row["microkwh"] for row in effective["payload"]["energy"])
                == 30000000
            )
            assert sum(row["minor"] for row in effective["payload"]["costs"]) == 2400
            assert service.closing.verify("shift")["verified"]
            assert not service.closing.drift("shift")["changed"]
            target = Path(folder) / "settlement.json"
            service.bundles.export("shift", target, effective=True)
            assert service.bundles.verify(target)["verified"]
            savings = verification(service)
            assert service.quality.system()["ok"]
            return {
                "status": "ok",
                "energy_microkwh": 30000000,
                "original_cost_minor": 2500,
                "corrected_cost_minor": 2400,
                "frozen_preserved": True,
                "verification": savings,
                "offline": True,
            }
