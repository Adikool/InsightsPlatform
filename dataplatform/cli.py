"""`insight` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

import pandas as pd

from .config import settings
from .errors import PlatformError
from .platform import Platform

WIDTH = 88


def _rule(title: str = "") -> str:
    if not title:
        return "-" * WIDTH
    return f"-- {title} " + "-" * max(0, WIDTH - len(title) - 4)


def _print_frame(df: pd.DataFrame, max_rows: int = 25) -> None:
    if df.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_rows", max_rows, "display.width", WIDTH, "display.max_columns", 20):
        print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"... {len(df):,} rows total")


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent + "  ")


# --------------------------------------------------------------------- source
def cmd_source_add(args) -> int:
    platform = Platform()
    options = dict(pair.split("=", 1) for pair in args.option or [])
    meta = platform.add_source(args.name, args.uri, type=args.type, **options)
    print(f"registered source {meta.name!r} ({meta.type})")
    objects = platform.list_objects(meta.name)
    print(f"{len(objects)} object(s) available: {', '.join(objects[:15])}")
    return 0


def cmd_source_list(args) -> int:
    platform = Platform()
    sources = platform.list_sources()
    if not sources:
        print("no sources registered — try: insight source add --name x --uri ...")
        return 0
    for meta in sources:
        print(f"{meta.name:<20} {meta.type:<8} {meta.uri}")
    return 0


def cmd_source_objects(args) -> int:
    for name in Platform().list_objects(args.name):
        print(name)
    return 0


def cmd_source_remove(args) -> int:
    Platform().remove_source(args.name)
    print(f"removed {args.name} and its datasets")
    return 0


# --------------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    platform = Platform()
    results = platform.ingest(args.source, obj=args.object, limit=args.limit)
    for result in results:
        print(result)
    ok = sum(1 for r in results if not r.dataset.startswith("!"))
    print(f"\n{ok}/{len(results)} object(s) ingested into {platform.warehouse.sqlalchemy_uri}")
    return 0


def cmd_datasets(args) -> int:
    platform = Platform()
    datasets = platform.list_datasets()
    if not datasets:
        print("nothing ingested yet")
        return 0
    for meta in datasets:
        print(f"{meta.name:<28} {meta.n_rows:>10,} rows  {len(meta.columns):>3} cols  <- {meta.source}:{meta.origin_object}")
    return 0


def cmd_describe(args) -> int:
    from .catalog import describe_dataset

    print(describe_dataset(Platform().dataset(args.dataset)))
    return 0


# ------------------------------------------------------------------------ ask
def cmd_ask(args) -> int:
    platform = Platform()

    if args.agent:
        result = platform.agent_ask(args.question)
        print(_rule("answer"))
        print(result.answer)
        if result.last_query:
            print(_rule("last query"))
            print(result.last_query.sql)
        return 0

    result = platform.ask(
        args.question,
        datasets=args.dataset,
        publish=args.publish,
        dashboard=args.dashboard,
        chart_name=args.chart_name,
    )

    print(_rule("interpretation"))
    print(_wrap(result.spec.explanation or "(none given)", indent=""))
    for warning in result.warnings:
        print(f"! {warning}")

    print(_rule("sql"))
    print(result.sql)

    print(_rule(f"result ({result.spec.chart})"))
    _print_frame(result.data, max_rows=args.max_rows)

    if result.published:
        print(_rule("superset"))
        print(f"chart:     {result.published.chart_url}")
        if result.published.dashboard_url:
            print(f"dashboard: {result.published.dashboard_url}")
        for note in result.published.notes:
            print(f"note: {note}")

    if args.json:
        print(_rule("spec"))
        print(result.spec.model_dump_json(indent=2))
    return 0


def cmd_dashboard(args) -> int:
    from .superset.layout import describe_layout

    from .nlp.heuristic import pick_dataset

    platform = Platform()
    if not platform.list_datasets():
        print("nothing ingested yet", file=sys.stderr)
        return 1
    # Same routing the NL layer uses, so the CLI default and the UI agree: the
    # request picks the dataset when it can, richness breaks the tie when it can't.
    datasets = args.dataset or [pick_dataset(args.request or "", platform.catalog).name]

    spec, compiled, published, problems = platform.build_dashboard(
        datasets, request=args.request or "", title=args.title or "", publish=not args.dry_run
    )

    print(_rule("plan"))
    print(f"{spec.title} — {len(compiled)} tile(s) from {', '.join(datasets)}")
    if spec.interpretation:
        print(_wrap(spec.interpretation, indent=""))
    print(_rule("layout"))
    print(describe_layout([tile for tile, _ in compiled]))

    for warning in problems:
        print(f"! {warning}")

    if args.show_sql:
        for tile, query in compiled:
            print(_rule(tile.title))
            print(query.sql)

    if published:
        print(_rule("superset"))
        print(f"dashboard: {published.dashboard_url}")
        print(f"charts:    {len(published.chart_ids)} created")
        for note in published.notes:
            print(f"note: {note}")
    else:
        print("\n(dry run — nothing published)")
    return 0


def cmd_sql(args) -> int:
    _print_frame(Platform().run_sql(args.statement), max_rows=args.max_rows)
    return 0


# -------------------------------------------------------------------- analyze
def cmd_analyze(args) -> int:
    platform = Platform()
    report = platform.analyze(
        args.dataset, target=args.target, sample=args.sample, narrate=args.narrate
    )

    if args.json:
        print(report.model_dump_json(indent=2))
        return 0

    stats = report.profile
    print(_rule("profile"))
    print(
        f"{stats.dataset}: {stats.n_rows:,} rows x {stats.n_columns} columns "
        f"({stats.memory_mb:.1f} MB in memory), {stats.n_duplicate_rows:,} duplicate rows"
    )
    print(f"  numeric:     {', '.join(stats.numeric_columns) or '-'}")
    print(f"  categorical: {', '.join(stats.categorical_columns) or '-'}")
    print(f"  temporal:    {', '.join(stats.temporal_columns) or '-'}")
    if stats.text_columns:
        print(f"  text:        {', '.join(stats.text_columns)}")
    print(f"  target: {report.target or '(none)'}    inferred task: {report.task}")

    print(_rule(f"patterns ({len(report.patterns)})"))
    if not report.patterns:
        print("nothing notable detected")
    for pattern in report.patterns:
        marker = {"strong": "!!", "notable": " !", "info": "  "}[pattern.severity]
        print(f"{marker} {pattern.kind}: {pattern.description}")
        if pattern.p_value is not None:
            print(f"     p = {pattern.p_value:.3g}")
        if pattern.implication and args.verbose:
            print(_wrap(f"-> {pattern.implication}", indent="     "))

    print(_rule(f"recommended algorithms ({len(report.recommendations)})"))
    for rec in report.recommendations:
        print(f"\n{rec.rank}. {rec.algorithm}   [{rec.task}, confidence: {rec.confidence}, {rec.library}]")
        for reason in rec.rationale:
            print(_wrap(f"* {reason}", indent="   "))
        if rec.preprocessing:
            print("   preprocessing:")
            for step in rec.preprocessing:
                print(_wrap(f"- {step}", indent="     "))
        if rec.evaluation:
            print(_wrap(f"evaluate with: {rec.evaluation}", indent="   "))
        for caveat in rec.caveats:
            print(_wrap(f"caveat: {caveat}", indent="   "))
        if args.code and rec.starter_code:
            print("   starter code:")
            for line in rec.starter_code.splitlines():
                print(f"     {line}")

    if report.narrative:
        print(_rule("summary"))
        print(textwrap.fill(report.narrative, width=WIDTH))
    return 0


def cmd_baseline(args) -> int:
    result = Platform().baseline(args.dataset, args.target, sample=args.sample)
    print(_rule("baseline"))
    print(f"{result.algorithm} on {result.n_train:,} rows, {result.n_features} features, {result.cv_folds}-fold CV")
    print(_rule("metrics"))
    for name, value in result.metrics.items():
        print(f"  {name:<32} {value: .4f}")
    if result.feature_importance:
        print(_rule("permutation importance (share)"))
        for name, value in result.feature_importance.items():
            bar = "#" * int(round(value * 40))
            print(f"  {name:<28} {value:6.1%} {bar}")
    for note in result.notes:
        print(f"note: {note}")
    return 0


# ------------------------------------------------------------------- superset
def cmd_superset_check(args) -> int:
    from .superset import check_connection

    me = check_connection()
    print(f"connected to {settings.superset_url} as {me.get('username', me)}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the web UI needs the api extra: pip install 'insight-platform[api]'",
            file=sys.stderr,
        )
        return 1

    print(f"Insight Platform UI -> http://{args.host}:{args.port}")
    print(f"  warehouse: {settings.warehouse_uri}")
    print(f"  language layer: {settings.model if settings.has_llm else 'keyword fallback (no API key)'}")
    uvicorn.run(
        "dataplatform.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def cmd_config(args) -> int:
    print(json.dumps(
        {
            "home": str(settings.home),
            "warehouse": settings.warehouse_uri,
            "catalog": str(settings.catalog_path),
            "model": settings.model,
            "effort": settings.effort,
            "llm_credentials": settings.has_llm,
            "superset_url": settings.superset_url,
            "max_rows": settings.max_rows,
        },
        indent=2,
    ))
    return 0


# ---------------------------------------------------------------------- parse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insight", description="connectors -> NL -> Superset -> data science"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    source = sub.add_parser("source", help="manage data sources").add_subparsers(
        dest="source_command", required=True
    )
    add = source.add_parser("add", help="register a source")
    add.add_argument("--name", required=True)
    add.add_argument("--uri", required=True, help="SQLAlchemy URI, or a path to a file/folder")
    add.add_argument("--type", choices=["sql", "excel", "csv", "parquet"])
    add.add_argument("--option", action="append", help="connector option, key=value")
    add.set_defaults(func=cmd_source_add)

    listing = source.add_parser("list", help="list sources")
    listing.set_defaults(func=cmd_source_list)

    objects = source.add_parser("objects", help="list readable objects in a source")
    objects.add_argument("name")
    objects.set_defaults(func=cmd_source_objects)

    remove = source.add_parser("remove", help="remove a source and its datasets")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_source_remove)

    ingest = sub.add_parser("ingest", help="load a source into the warehouse")
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--object", help="one table/sheet; omit for everything")
    ingest.add_argument("--all", action="store_true", help="(default) ingest every object")
    ingest.add_argument("--limit", type=int, help="row cap per object")
    ingest.set_defaults(func=cmd_ingest)

    datasets = sub.add_parser("datasets", help="list ingested datasets")
    datasets.set_defaults(func=cmd_datasets)

    describe = sub.add_parser("describe", help="show a dataset's schema as the model sees it")
    describe.add_argument("dataset")
    describe.set_defaults(func=cmd_describe)

    ask = sub.add_parser("ask", help="ask a question in plain English")
    ask.add_argument("question")
    ask.add_argument("--dataset", action="append", help="restrict to these datasets")
    ask.add_argument("--agent", action="store_true", help="use the multi-step analyst agent")
    ask.add_argument("--publish", action="store_true", help="publish to Superset")
    ask.add_argument("--dashboard", help="dashboard title to pin the chart onto")
    ask.add_argument("--chart-name", dest="chart_name")
    ask.add_argument("--max-rows", type=int, default=25)
    ask.add_argument("--json", action="store_true", help="also print the QuerySpec")
    ask.set_defaults(func=cmd_ask)

    dashboard = sub.add_parser("dashboard", help="compose a multi-tile Superset dashboard")
    dashboard.add_argument("request", nargs="?", default="", help="what the dashboard should show")
    dashboard.add_argument(
        "--dataset", action="append",
        help="dataset to build from; repeat for a multi-table dashboard (default: the best match)",
    )
    dashboard.add_argument("--title", help="dashboard title")
    dashboard.add_argument("--dry-run", action="store_true", help="plan only, publish nothing")
    dashboard.add_argument("--show-sql", action="store_true", help="print each tile's SQL")
    dashboard.set_defaults(func=cmd_dashboard)

    sql = sub.add_parser("sql", help="run read-only SQL through the guard")
    sql.add_argument("statement")
    sql.add_argument("--max-rows", type=int, default=25)
    sql.set_defaults(func=cmd_sql)

    analyze = sub.add_parser("analyze", help="profile, detect patterns, recommend algorithms")
    analyze.add_argument("--dataset", required=True)
    analyze.add_argument("--target", help="column to predict; omit for unsupervised advice")
    analyze.add_argument("--sample", type=int, help="row cap for the analysis")
    analyze.add_argument("--narrate", action="store_true", help="add an LLM executive summary")
    analyze.add_argument("--code", action="store_true", help="print starter code per recommendation")
    analyze.add_argument("--verbose", action="store_true", help="print each pattern's implication")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    baseline = sub.add_parser("baseline", help="actually fit the top recommendation")
    baseline.add_argument("--dataset", required=True)
    baseline.add_argument("--target", required=True)
    baseline.add_argument("--sample", type=int)
    baseline.set_defaults(func=cmd_baseline)

    superset = sub.add_parser("superset", help="Superset operations").add_subparsers(
        dest="superset_command", required=True
    )
    check = superset.add_parser("check", help="verify credentials and reachability")
    check.set_defaults(func=cmd_superset_check)

    serve = sub.add_parser("serve", help="run the web UI and REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    serve.set_defaults(func=cmd_serve)

    config = sub.add_parser("config", help="show resolved configuration")
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
