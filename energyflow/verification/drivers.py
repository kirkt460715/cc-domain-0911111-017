"""External driver series (production, runtime, temperature) with revisions.

Driver definitions are immutable. Entries hold exact decimal values over
half-open intervals and are corrected through a revision chain, mirroring
meter readings, so later data repairs stay auditable.
"""
from ..values import (
    key,
    text,
    instant,
    integer,
    fixed,
    canonical,
    fingerprint,
    parse_json,
    reject,
    object_fields,
)
from ..calendar import coverage_summary


class Drivers:
    def __init__(self, store, catalog):
        self.store = store
        self.catalog = catalog

    def define(self, driver, meter, name, aggregation, unit=""):
        key(driver)
        self.catalog.get(meter)
        name = text(name, 300)
        if aggregation not in {"sum", "mean"}:
            reject("input", "driver aggregation must be sum or mean")
        unit = text(unit, 100, True)
        payload = {
            "id": driver,
            "meter": meter,
            "name": name,
            "aggregation": aggregation,
            "unit": unit,
        }
        with self.store.atomic() as con:
            old = self.store.one("SELECT id FROM drivers WHERE id=?", (driver,))
            if old:
                if self.get(driver) != payload:
                    reject("conflict", "driver definition is immutable")
                return payload
            con.execute(
                "INSERT INTO drivers VALUES(?,?,?)",
                (driver, canonical(payload), fingerprint(payload)),
            )
            self.store.event("driver_defined", driver, payload)
        return payload

    def get(self, driver):
        return self.store.payload("drivers", driver)[1]

    def list(self, meter=None):
        rows = [
            self.get(row["id"])
            for row in self.store.rows("SELECT id FROM drivers ORDER BY id")
        ]
        if meter is not None:
            rows = [row for row in rows if row["meter"] == meter]
        return rows

    def normalize(self, data):
        object_fields(
            data,
            {"id", "driver", "start", "end", "value", "note"},
            {"id", "driver", "start", "end", "value"},
        )
        ident = key(data["id"])
        definition = self.get(key(data["driver"]))
        start = instant(data["start"])
        end = instant(data["end"])
        if end <= start or end - start > 366 * 86400:
            reject("time", "invalid driver entry interval")
        signed = definition["aggregation"] == "mean"
        value = fixed(data["value"], signed=signed)
        note = text(data.get("note", ""), 1000, True)
        return {
            "id": ident,
            "driver": definition["id"],
            "start": start,
            "end": end,
            "value": value,
            "note": note,
        }

    def put(self, entry, driver, start, end, value, note=""):
        normalized = self.normalize(
            {
                "id": entry,
                "driver": driver,
                "start": start,
                "end": end,
                "value": value,
                "note": note,
            }
        )
        checksum = fingerprint(normalized)
        with self.store.atomic() as con:
            old = self.store.one(
                "SELECT id FROM driver_entries WHERE id=?", (normalized["id"],)
            )
            if old:
                first = self.store.one(
                    "SELECT payload,digest FROM driver_history WHERE id=? AND revision=1",
                    (normalized["id"],),
                )
                if first["digest"] != checksum:
                    reject("conflict", "driver entry identity differs from ingestion")
                return {**self.get_entry(normalized["id"]), "reused": True}
            overlap = self.store.one(
                "SELECT id FROM driver_entries WHERE driver=? AND start<? AND end>?",
                (normalized["driver"], normalized["end"], normalized["start"]),
            )
            if overlap:
                reject("overlap", "driver entries overlap", existing=overlap["id"])
            con.execute(
                "INSERT INTO driver_entries VALUES(?,?,?,?,1,?,?)",
                (
                    normalized["id"],
                    normalized["driver"],
                    normalized["start"],
                    normalized["end"],
                    canonical(normalized),
                    checksum,
                ),
            )
            con.execute(
                "INSERT INTO driver_history VALUES(?,1,?,?,?)",
                (normalized["id"], canonical(normalized), checksum, "initial entry"),
            )
            self.store.event(
                "driver_entry_added", normalized["id"], {"digest": checksum}
            )
        return {**self.get_entry(normalized["id"]), "reused": False}

    def get_entry(self, ident, revision=None):
        if revision is None:
            row, payload = self.store.payload("driver_entries", ident)
            return {**payload, "revision": row["revision"], "digest": row["digest"]}
        integer(revision, 1)
        row = self.store.one(
            "SELECT * FROM driver_history WHERE id=? AND revision=?", (ident, revision)
        )
        if row is None:
            reject("missing", "driver entry revision not found")
        payload = parse_json(row["payload"])
        if fingerprint(payload) != row["digest"]:
            reject("integrity", "driver entry history differs")
        return {
            **payload,
            "revision": revision,
            "digest": row["digest"],
            "reason": row["reason"],
        }

    def correct(self, entry, expected_revision, value, reason, note=""):
        integer(expected_revision, 1)
        reason = text(reason)
        with self.store.atomic() as con:
            current = self.get_entry(entry)
            if current["revision"] != expected_revision:
                reject("conflict", "driver entry was already corrected")
            proposal = {
                name: current[name]
                for name in ["id", "driver", "start", "end", "value", "note"]
            }
            proposal.update(value=value, note=note if note else current["note"])
            proposal = self.normalize(proposal)
            if fingerprint(proposal) == current["digest"]:
                reject("input", "correction must change the driver entry")
            revision = expected_revision + 1
            checksum = fingerprint(proposal)
            con.execute(
                "UPDATE driver_entries SET revision=?,payload=?,digest=? WHERE id=?",
                (revision, canonical(proposal), checksum, entry),
            )
            con.execute(
                "INSERT INTO driver_history VALUES(?,?,?,?,?)",
                (entry, revision, canonical(proposal), checksum, reason),
            )
            self.store.event(
                "driver_entry_corrected",
                entry,
                {
                    "revision": revision,
                    "previous": current["digest"],
                    "digest": checksum,
                    "reason": reason,
                },
            )
        return self.get_entry(entry)

    def history(self, entry):
        self.get_entry(entry)
        return [
            self.get_entry(entry, row["revision"])
            for row in self.store.rows(
                "SELECT revision FROM driver_history WHERE id=? ORDER BY revision",
                (entry,),
            )
        ]

    def series(self, driver, start, end):
        definition = self.get(driver)
        start = instant(start)
        end = instant(end)
        if end <= start:
            reject("time", "invalid half-open interval")
        return [
            {**self.get_entry(row["id"]), "definition": definition["aggregation"]}
            for row in self.store.rows(
                "SELECT id FROM driver_entries WHERE driver=? AND start<? AND end>? ORDER BY start,id",
                (driver, end, start),
            )
        ]

    def coverage(self, driver, start, end):
        rows = self.series(driver, start, end)
        report = coverage_summary(
            start, end, [(row["start"], row["end"]) for row in rows]
        )
        return {**report, "driver": driver, "entry_digest": fingerprint(rows)}
