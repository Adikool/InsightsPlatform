"""Raw-SQL guard, for the `--allow-raw-sql` escape hatch only.

The primary path never needs this: specs compile to SQL we built ourselves. But
analysts do sometimes want to paste a query, so when raw SQL is permitted it is
checked here first. This is a *deny* gate, not a sanitiser — anything it is not
sure about is rejected.
"""

from __future__ import annotations

import re

from ..catalog import Catalog
from ..config import settings
from ..errors import QueryValidationError

_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "grant",
    "revoke", "merge", "replace", "attach", "detach", "copy", "export", "import",
    "install", "load", "call", "vacuum", "pragma", "reset", "begin",
    "commit", "rollback", "execute",
}
# "set" is checked separately as a leading keyword to avoid false positives
# on column names that contain the substring "set" (e.g. dataset, offset, reset_flag).
_FORBIDDEN_LEADING = {"set"}
_FORBIDDEN_FUNCS = {"read_csv", "read_parquet", "read_json", "pg_read_file", "load_extension"}

_COMMENT = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.S)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([\"`\[]?[A-Za-z_][A-Za-z0-9_.]*[\"`\]]?)", re.I)


def _strip_comments(sql: str) -> str:
    return _COMMENT.sub(" ", sql)


def validate_sql(sql: str, catalog: Catalog, max_rows: int | None = None) -> str:
    """Return an executable statement or raise. Injects a LIMIT if absent."""
    if not sql or not sql.strip():
        raise QueryValidationError("empty statement")

    body = _strip_comments(sql).strip().rstrip(";")

    if ";" in body:
        raise QueryValidationError("only a single statement is allowed")

    lowered = body.lower()
    first = lowered.split(None, 1)[0] if lowered.split() else ""
    if first not in ("select", "with"):
        raise QueryValidationError(f"only SELECT/WITH statements are allowed, got {first.upper()!r}")

    words = set(re.findall(r"\b[a-z_]+\b", lowered))
    banned = words & _FORBIDDEN
    # Check leading-keyword-only terms separately to avoid false positives on
    # column/table names containing those substrings (e.g. "dataset", "offset").
    if first in _FORBIDDEN_LEADING:
        banned.add(first)
    if banned:
        raise QueryValidationError(f"statement contains forbidden keyword(s): {sorted(banned)}")
    banned_funcs = words & _FORBIDDEN_FUNCS
    if banned_funcs:
        raise QueryValidationError(f"statement calls forbidden function(s): {sorted(banned_funcs)}")

    known = {d.name.lower() for d in catalog.list_datasets()}
    cte_names = {m.lower() for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", body, re.I)}
    referenced = {
        ref.strip('"`[]').split(".")[-1].lower() for ref in _TABLE_REF.findall(body)
    }
    unknown = referenced - known - cte_names
    if unknown:
        raise QueryValidationError(
            f"statement references table(s) not in the catalog: {sorted(unknown)}"
        )

    cap = max_rows or settings.max_rows
    if not re.search(r"\blimit\s+\d+", lowered):
        body += f"\nLIMIT {cap}"
    return body
