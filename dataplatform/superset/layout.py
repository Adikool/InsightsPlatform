"""Build Superset's `position_json`.

Superset stores a dashboard layout as a flat dict of nodes keyed by id, linked by
`children` and `parents`, with a fixed spine: ROOT -> GRID -> ROW -> CHART. Widths
are in twelfths; heights in ~8px units. Getting `parents` wrong is the usual cause
of a dashboard that saves fine and then renders blank, so every node here carries
its full ancestor chain.

Charts are packed into rows greedily so a row never exceeds 12 columns — which is
what makes four KPI cards sit side by side and the detail table span the full width
underneath.
"""

from __future__ import annotations

from typing import Any

from ..reports.spec import GRID_COLUMNS, Tile

ROOT_ID = "ROOT_ID"
GRID_ID = "GRID_ID"
HEADER_ID = "HEADER_ID"


def pack_rows(tiles: list[Tile]) -> list[list[Tile]]:
    """Greedy left-to-right packing, preserving order.

    Order is preserved rather than optimised: the composer already sorted tiles
    into reading order (KPIs, trend, breakdowns, detail), and re-ordering to fill
    rows more tightly would put the detail table somewhere other than the bottom.
    """
    rows: list[list[Tile]] = []
    current: list[Tile] = []
    used = 0

    for tile in tiles:
        width = max(1, min(int(tile.width or 6), GRID_COLUMNS))
        if used + width > GRID_COLUMNS and current:
            rows.append(current)
            current, used = [], 0
        current.append(tile)
        used += width

    if current:
        rows.append(current)
    return rows


def build_position_json(
    title: str, tiles: list[Tile], chart_ids: list[int]
) -> dict[str, Any]:
    """Lay out `tiles` (already ordered) against their created `chart_ids`."""
    if len(tiles) != len(chart_ids):
        raise ValueError("every tile needs exactly one chart id")

    paired = list(zip(tiles, chart_ids))
    rows = pack_rows([tile for tile, _ in paired])

    position: dict[str, Any] = {
        "DASHBOARD_VERSION_KEY": "v2",
        ROOT_ID: {"type": "ROOT", "id": ROOT_ID, "children": [GRID_ID]},
        HEADER_ID: {"type": "HEADER", "id": HEADER_ID, "meta": {"text": title}},
        GRID_ID: {"type": "GRID", "id": GRID_ID, "children": [], "parents": [ROOT_ID]},
    }

    ids_by_tile = {id(tile): chart_id for tile, chart_id in paired}
    row_number = 0

    for row in rows:
        row_number += 1
        row_id = f"ROW-insight-{row_number}"
        position[GRID_ID]["children"].append(row_id)
        position[row_id] = {
            "type": "ROW",
            "id": row_id,
            "children": [],
            "parents": [ROOT_ID, GRID_ID],
            "meta": {"background": "BACKGROUND_TRANSPARENT"},
        }

        for index, tile in enumerate(row, start=1):
            chart_key = f"CHART-insight-{row_number}-{index}"
            position[row_id]["children"].append(chart_key)
            position[chart_key] = {
                "type": "CHART",
                "id": chart_key,
                "children": [],
                "parents": [ROOT_ID, GRID_ID, row_id],
                "meta": {
                    "chartId": ids_by_tile[id(tile)],
                    "width": max(1, min(int(tile.width or 6), GRID_COLUMNS)),
                    "height": int(tile.height or 50),
                    "sliceName": tile.title,
                },
            }

    return position


def describe_layout(tiles: list[Tile]) -> str:
    """Human-readable summary of the grid, for the CLI and for tests."""
    lines = []
    for number, row in enumerate(pack_rows(tiles), start=1):
        cells = ", ".join(f"{tile.title} [{tile.width}/12, {tile.role}]" for tile in row)
        lines.append(f"  row {number}: {cells}")
    return "\n".join(lines)
