"""Build a small sales report with two Tools and structured output."""

from __future__ import annotations

import asyncio
import math
import os
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field

from moduagent import Agent, ConsoleEventSink, VLLMClient, tool


SALES = (
    {"date": "2025-01-08", "region": "north", "amount": 1200.0},
    {"date": "2025-01-19", "region": "south", "amount": 900.0},
    {"date": "2025-02-07", "region": "north", "amount": 1500.0},
    {"date": "2025-02-21", "region": "south", "amount": 1300.0},
    {"date": "2025-03-04", "region": "north", "amount": 1700.0},
    {"date": "2025-03-23", "region": "south", "amount": 1600.0},
)

ARTIFACT_DIR = Path(
    os.getenv(
        "MODUAGENT_ARTIFACT_DIR",
        Path(__file__).resolve().parent / "artifacts",
    )
).expanduser()


class SalesReport(BaseModel):
    title: str
    period: str
    summary: str
    total_sales: float = Field(ge=0)
    top_month: str
    chart_path: str


@tool
def query_sales(
    start_date: str,
    end_date: str,
    region: str = "all",
) -> dict[str, object]:
    """Return monthly sales totals for an inclusive ISO date range and region."""

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    normalized_region = region.strip().lower()
    if normalized_region not in {"all", "north", "south"}:
        raise ValueError("region must be all, north, or south")

    monthly: dict[str, float] = {}
    for sale in SALES:
        sold_on = date.fromisoformat(str(sale["date"]))
        if not start <= sold_on <= end:
            continue
        if normalized_region != "all" and sale["region"] != normalized_region:
            continue
        month = sold_on.strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0.0) + float(sale["amount"])

    rows = [
        {"month": month, "sales": round(amount, 2)}
        for month, amount in sorted(monthly.items())
    ]
    return {
        "period": f"{start.isoformat()} to {end.isoformat()}",
        "region": normalized_region,
        "rows": rows,
        "total_sales": round(sum(monthly.values()), 2),
    }


@tool
def plot_graph(
    labels: list[str],
    values: list[float],
    title: str = "Monthly sales",
) -> dict[str, object]:
    """Create an SVG bar chart and return its file path."""

    if not labels or len(labels) != len(values):
        raise ValueError("labels and values must have the same non-zero length")
    if len(labels) > 24:
        raise ValueError("at most 24 chart points are supported")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("values must be finite and non-negative")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = ARTIFACT_DIR / "sales_report.svg"
    chart_path.write_text(
        _bar_chart_svg(labels=labels, values=values, title=title),
        encoding="utf-8",
    )
    return {"chart_path": str(chart_path), "points": len(labels)}


def _bar_chart_svg(*, labels: list[str], values: list[float], title: str) -> str:
    width = 720
    height = 420
    left = 70
    top = 70
    bottom = 70
    chart_height = height - top - bottom
    slot_width = (width - left - 30) / len(values)
    bar_width = slot_width * 0.62
    maximum = max(values) or 1.0

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
            f'font-family="sans-serif" font-size="22">{escape(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
            'stroke="#334155"/>'
        ),
        (
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - 20}" '
            f'y2="{height - bottom}" stroke="#334155"/>'
        ),
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = chart_height * value / maximum
        x = left + index * slot_width + (slot_width - bar_width) / 2
        y = top + chart_height - bar_height
        elements.extend(
            (
                (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{bar_height:.1f}" fill="#2563eb"/>'
                ),
                (
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="12">'
                    f"{value:,.0f}</text>"
                ),
                (
                    f'<text x="{x + bar_width / 2:.1f}" y="{height - bottom + 24}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="12">'
                    f"{escape(label)}</text>"
                ),
            )
        )

    elements.append("</svg>")
    return "\n".join(elements)


def build_agent(model, *, event_sink=None):
    return Agent.create(
        model=model,
        instructions=(
            "Create concise sales reports from verified Tool results. First call "
            "query_sales with the requested filters. Then call plot_graph with "
            "the returned months and sales values. Never invent data or a chart "
            "path. Finish with the requested SalesReport."
        ),
        tools=[query_sales, plot_graph],
        execution="standard",
        output=SalesReport,
        event_sink=event_sink,
    )


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 512},
    ) as model:
        agent = build_agent(model, event_sink=ConsoleEventSink())
        report = await agent.ask(
            "Report sales from 2025-01-01 through 2025-03-31 for all regions."
        )
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
