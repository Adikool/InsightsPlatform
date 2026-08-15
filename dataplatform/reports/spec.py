"""Dashboard composition types.

`QuerySpec` answers one question. A dashboard is several questions arranged on a
grid, so it needs its own shape: a list of tiles, each carrying a QuerySpec plus
where it sits. Keeping layout out of QuerySpec matters — the same query should be
publishable standalone or as one tile of a report without changing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..nlp.spec import QuerySpec

TileRole = Literal["kpi", "trend", "breakdown", "detail", "distribution"]

# Superset's grid is 12 columns wide; height is in ~8px row units.
GRID_COLUMNS = 12
KPI_HEIGHT = 24
CHART_HEIGHT = 50
TABLE_HEIGHT = 60


class Tile(BaseModel):
    spec: QuerySpec
    role: TileRole = "breakdown"
    width: int = Field(default=6, description="Grid columns out of 12.")
    height: int = Field(default=CHART_HEIGHT)

    @property
    def title(self) -> str:
        return self.spec.title or f"{self.spec.dataset} view"


class DashboardSpec(BaseModel):
    title: str
    description: str = ""
    tiles: list[Tile] = Field(default_factory=list)
    # Plain-English readback of what was built and which parts of the request were
    # honoured, so a misreading shows on the page instead of in a wrong chart.
    interpretation: str = ""

    def by_role(self, role: TileRole) -> list[Tile]:
        return [tile for tile in self.tiles if tile.role == role]

    @property
    def ordered(self) -> list[Tile]:
        """KPIs first, then trends, breakdowns, distributions, detail last.

        The detail table goes at the bottom by construction — that is the
        conventional reading order for a report, and it is what people mean by
        'a detail table below the graphs'.
        """
        order = {"kpi": 0, "trend": 1, "breakdown": 2, "distribution": 3, "detail": 4}
        return sorted(self.tiles, key=lambda t: order.get(t.role, 2))


class DashboardPlan(BaseModel):
    """What the language layer returns: specs without layout.

    Widths are assigned by the composer from each tile's role rather than by the
    model, because a model asked to pick grid columns will happily emit tiles that
    do not add up to a row.
    """

    title: str = Field(description="Short dashboard title.")
    description: str = Field(default="", description="One line on what it shows.")
    tiles: list[Tile] = Field(description="4-8 tiles covering KPIs, trend, breakdowns, detail.")
