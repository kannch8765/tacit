"""The five card buckets: required fields, allowed values and lifecycle policy.

Validation here is code, not a prompt: a proposal that doesn't fit its bucket is rejected with a reason code.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Bucket:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    # Lifecycle: "supersede" = a newer card on the same lineage replaces the old version;
    #            "reinforce" = seeing it again strengthens the existing card.
    on_repeat: str = "supersede"
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Evidence roles that may ground a card in this bucket
    evidence_roles: tuple[str, ...] = ("user", "assistant", "tool")


BUCKETS = {b.name: b for b in [
    Bucket("preference", required=("who",), on_repeat="supersede"),
    Bucket("workflow_habit", required=("trigger",), on_repeat="reinforce"),
    # A knowledge-cutoff patch. The model can't be the source of facts it doesn't know, so only the user or a tool
    # result (web search, fetched page) may ground it.
    Bucket("world_update", required=("source",), optional=("replaces", "valid_from"), on_repeat="supersede",
           choices={"source": ("user", "tool")}, evidence_roles=("user", "tool")),
    Bucket("decision", required=("rationale",), on_repeat="supersede"),
    Bucket("gotcha", required=("symptom", "fix"), on_repeat="reinforce"),
]}


class Rejected(ValueError):
    """A proposal that can't be stored. `code` is a stable reason code for the outcome ledger."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


PLACEHOLDERS = {"n/a", "na", "none", "null", "nil", "-", "--", "unknown", "not applicable", "tbd"}


def validate_fields(bucket_name, fields):
    """Return the cleaned fields for a bucket, or raise Rejected."""
    bucket = BUCKETS.get(bucket_name)
    if bucket is None:
        raise Rejected("unknown_bucket", bucket_name)
    if not isinstance(fields, dict):
        raise Rejected("bad_fields", "fields must be an object")
    allowed = set(bucket.required) | set(bucket.optional)
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise Rejected("unknown_field", ", ".join(unknown))
    clean = {}
    for name in allowed:
        value = fields.get(name)
        if value is None or (isinstance(value, str) and value.strip().strip(".").lower() in PLACEHOLDERS | {""}):
            if name in bucket.required:
                raise Rejected("missing_field", name)
            continue
        if not isinstance(value, str):
            raise Rejected("bad_field_type", name)
        if name in bucket.choices and value not in bucket.choices[name]:
            raise Rejected("bad_field_value", f"{name}={value}")
        clean[name] = value.strip()
    return clean
