"""DivineLines command line.

    divinelines migrate      import the legacy database into the v2 schema
    divinelines refresh      fetch results, fixtures, injuries and odds
    divinelines train        train and register models
    divinelines scan         generate predictions and +EV opportunities
    divinelines backtest     walk-forward evaluation
    divinelines ablate       feature-ablation study
    divinelines settle       resolve closing lines, compute CLV, grade results
    divinelines evaluate     realised performance from the ledger
    divinelines status       data, source and model health
    divinelines health       predictive vs betting health, by sport and window
    divinelines clv          closing-line value: distribution, cohorts, skill test
    divinelines lineups      fetch lineups for upcoming (or historical) fixtures
    divinelines backfill     historical odds / ESPN id mapping
    divinelines replay       replay a season into the ledger at opening prices
    divinelines serve        run the API
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import pandas as pd

from .config import settings
from .logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def _print(payload: Any) -> None:
    if isinstance(payload, pd.DataFrame):
        print(payload.to_string(index=False) if not payload.empty else "(no rows)")
    else:
        print(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_migrate(args: argparse.Namespace) -> int:
    from .db.migrate import migrate_legacy_nba, migration_summary

    stats = migrate_legacy_nba(strict=not args.lenient)
    _print(stats)
    _print(migration_summary())
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    from .pipeline.refresh import refresh_all

    report = refresh_all(sports=args.sport, include_odds=not args.no_odds,
                         include_match_detail=not args.no_match_detail,
                         soccer_leagues=args.league)
    _print(report.to_dict())
    return 0 if report.to_dict()["ok"] else 1


def cmd_train(args: argparse.Namespace) -> int:
    from .pipeline.train import train_nba, train_soccer

    outcomes = []
    if "nba" in args.sport:
        outcomes.append(train_nba(variant=args.variant).to_dict())
    if "soccer" in args.sport:
        outcomes.append(train_soccer(variant=args.variant or "full",
                                     use_market=args.use_market).to_dict())
    _print(outcomes)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from .pipeline.predict import scan

    results = scan(args.sport, paper_trade=args.paper, days_ahead=args.days,
                   persist=not args.dry_run)
    for sport, payload in results.items():
        if "error" in payload:
            print(f"[{sport}] error: {payload['error']}")
            continue
        opportunities = payload["opportunities"]
        print(f"\n[{sport}] {len(payload['predictions'])} predictions, "
              f"{len(opportunities)} qualifying opportunities")
        for note in payload.get("notes", []):
            print(f"  note: {note}")
        for warning in payload.get("warnings", []):
            print(f"  warning: {warning}")
        for opportunity in opportunities[: args.limit]:
            print(
                f"  {opportunity['game_date']} {opportunity['away_name']} @ "
                f"{opportunity['home_name']} | {opportunity['selection']:5s} "
                f"model {opportunity['model_probability']:.3f} vs market "
                f"{(opportunity['market_probability'] or 0):.3f} | edge "
                f"{(opportunity['edge'] or 0):+.3f} | {opportunity['price_decimal']} "
                f"({opportunity['bookmaker']}) | score {opportunity['edge_score']:.1f}/10 "
                f"| stake {opportunity['stake']:.2f}"
            )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    summaries: dict[str, Any] = {}

    if "nba" in args.sport:
        from .backtest.nba_backtest import run_nba_backtest

        result = run_nba_backtest(variant=args.variant, seasons=args.seasons)
        summaries["nba"] = result.summary()
        print("\n=== NBA (probability quality, walk-forward by season) ===")
        for name, metrics in result.probability_metrics.items():
            print(f"  {name:20s} logloss={metrics['log_loss']:.4f} "
                  f"brier={metrics['brier']:.4f} acc={metrics['accuracy']:.3f} "
                  f"ece={metrics['ece']:.4f} skill={metrics['brier_skill']:+.4f}")
        print(f"  note: {result.note}")

    if "soccer" in args.sport:
        from .backtest.soccer_backtest import run_soccer_backtest

        result = run_soccer_backtest(args.league, variant=args.variant,
                                     seasons=args.seasons, use_market=args.use_market,
                                     price_book=args.price_book)
        summaries["soccer"] = result.summary()
        metrics = result.metrics.to_dict()
        print("\n=== Soccer (walk-forward, real historical prices) ===")
        print(f"  model  : logloss={result.probability_metrics['model']['log_loss']:.5f}")
        market = result.probability_metrics.get("market_novig")
        if market:
            print(f"  market : logloss={market['log_loss']:.5f} (no-vig benchmark)")
        print(f"  bets={metrics['bets']} roi={metrics['roi']:+.4f} "
              f"profit={metrics['profit']:+.2f} clv={metrics['clv_mean_pct']}")
        if not result.by_price.empty:
            _print(result.by_price)

    if args.save:
        for sport, summary in summaries.items():
            path = settings.paths.artifacts_dir / f"{sport}_backtest_summary.json"
            path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
            print(f"\nsaved {path}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from .models.experiments import run_ablation

    frame = run_ablation(args.sport[0], seasons=args.seasons)
    _print(frame)
    return 0


def cmd_settle(args: argparse.Namespace) -> int:
    from .betting.closing_line import ClosingLinePolicy
    from .betting.ledger import settle_open_bets
    from .betting.settlement import settle

    policy = ClosingLinePolicy(
        cutoff_seconds=args.cutoff, aggregation=args.close_aggregation,
        bookmaker=args.close_book,
    )
    report = settle(sport=args.sport, day=args.date, policy=policy,
                    dry_run=args.dry_run, full_reconcile=args.full)
    data = report.to_dict()

    print("DivineLines Settlement")
    print("-" * 46)
    print(f"  Scanned:         {data['scanned']}")
    print(f"  CLV records:     {data['created']}{'  (dry run)' if args.dry_run else ''}")
    print(f"  Close found:     {data['close_found']}")
    print(f"  Awaiting close:  {data['awaiting_close']}")
    print(f"  No close found:  {data['no_close']}")
    print(f"  Results graded:  {data['results_settled']}")
    print(f"  Awaiting result: {data['awaiting_result']}")
    if data["invalid"]:
        print(f"  Invalid:         {data['invalid']}")

    clv = data.get("clv") or {}
    if clv.get("n"):
        print()
        print(f"  CLV basis:       {clv.get('basis', 'n/a')}")
        print(f"  Sample:          {clv['n']}")
        print(f"  Mean CLV:        {clv['mean_clv_price_pct']:+.2f}%")
        print(f"  Median CLV:      {clv['median_clv_price_pct']:+.2f}%")
        print(f"  Positive CLV:    {clv['beat_close_rate']:.1%}")
        if clv.get("ci_low") is not None:
            print(f"  95% CI:          [{clv['ci_low']:+.2f}%, {clv['ci_high']:+.2f}%]")
        print(f"  Reading:         {clv['interpretation']}")
        same = clv.get("same_book")
        if same and same.get("n"):
            print(f"  Same-book CLV:   {same['mean_clv_price_pct']:+.2f}% (n={same['n']})")
    else:
        print("\n  CLV:             no settled records yet")

    if data.get("paper_roi") is not None:
        print(f"\n  Paper ROI:       {data['paper_roi']:+.2%} "
              f"(profit {data['paper_profit']:+.2f})")
        print("  Note:            CLV and ROI measure different things; "
              "neither implies the other.")

    if data["error_count"]:
        print(f"\n  Errors:          {data['error_count']}")
        for error in data["errors"]:
            print(f"    - {error}")

    # The V2 bet ledger is still graded so older performance views keep working.
    if not args.dry_run:
        settle_open_bets()
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from .analytics.model_health import (
        WINDOWS,
        compute_health,
        detect_regression,
        persist_snapshot,
    )

    windows = dict(WINDOWS)
    for sport in args.sport:
        print(f"\n=== {sport.upper()} model health ===")
        for label in (args.window or list(windows)):
            result = compute_health(sport, market=args.market, window_label=label,
                                    window_days=windows.get(label))
            predictive = result.predictive or {}
            comparison = result.market_comparison or {}
            betting = result.betting or {}
            print(f"\n  [{label}] n={result.sample_size}  status={result.status}")
            print(f"    {result.status_reason}")
            if predictive:
                print(f"    Brier {predictive['brier']:.4f} | "
                      f"log loss {predictive['log_loss']:.4f} | "
                      f"ECE {predictive['ece']:.4f} | skill {predictive['brier_skill']:+.4f}")
            if comparison:
                verdict = "model" if comparison["beats_market"] else "market"
                print(f"    vs market: model {comparison['model_log_loss']:.4f} | "
                      f"market {comparison['market_log_loss']:.4f} -> {verdict} better "
                      f"(n={comparison['n']})")
            clv = betting.get("clv") or {}
            if clv.get("n"):
                print(f"    CLV {clv['mean_clv_price_pct']:+.2f}% "
                      f"(n={clv['n']}, {clv['interpretation']})")
            if betting.get("roi") is not None:
                print(f"    ROI {betting['roi']:+.2%} on {betting['settled_bets']} settled")
            if args.persist:
                persist_snapshot(result)

        regression = detect_regression(sport)
        if regression.get("regression"):
            print(f"\n  !! MODEL REGRESSION: {regression['reason']}")
        else:
            print(f"\n  regression check: {regression.get('reason')}")
    return 0


def cmd_clv(args: argparse.Namespace) -> int:
    from .analytics.clv_analysis import clv_report
    from .analytics.clv_skill import decompose_clv

    report = clv_report(args.sport, basis=args.basis)
    overall = report["overall"]
    print(f"CLV — {report['basis_description']}")
    print("-" * 62)
    if not overall["n"]:
        print("  no CLV records yet; run `divinelines settle` after events finish")
        return 0

    print(f"  Sample:        {overall['n']}")
    print(f"  Mean:          {overall['mean_clv_price_pct']:+.3f}%")
    print(f"  Median:        {overall['median_clv_price_pct']:+.3f}%")
    print(f"  Positive rate: {overall['beat_close_rate']:.1%}")
    if overall.get("ci_low") is not None:
        print(f"  95% CI:        [{overall['ci_low']:+.3f}%, {overall['ci_high']:+.3f}%]")
    print(f"  Reading:       {overall['interpretation']}")

    if args.cohort:
        table = report["cohorts"].get(args.cohort)
        if table:
            print(f"\n  By {args.cohort}:")
            _print(pd.DataFrame(table))
        else:
            print(f"\n  no cohort data for '{args.cohort}'")

    if args.skill:
        decomposition = decompose_clv(args.sport or "nba", mode=args.mode)
        if not decomposition.get("available"):
            print(f"\n  skill test unavailable: {decomposition.get('reason')}")
            return 0
        components = decomposition["components"]
        print("\n  Where the CLV comes from:")
        print(f"    reported mean:      {components['reported_clv_mean_pct']:+.3f}%")
        print(f"    line shopping:      {components['line_shopping_pct']:+.3f}%")
        print(f"    market drift:       {components['market_drift_pct']:+.3f}%")
        control = decomposition["both_sides_control"]
        print(f"    both-sides control: {control.get('mean_of_both_sides_pct')}% "
              f"over {control.get('n_games')} games")
        for name, test in decomposition["skill_tests"].items():
            print(f"\n    [{name}] {test['basis']}")
            print(f"      recommended n={test['recommended']['n']} "
                  f"mean={test['recommended']['mean']:+.3f}%")
            print(f"      rejected    n={test['rejected']['n']} "
                  f"mean={test['rejected']['mean']:+.3f}%")
            print(f"      difference  {test['difference']:+.3f}pp "
                  f"CI [{test['ci_low']}, {test['ci_high']}] "
                  f"significant={test['significant']}")
            print(f"      -> {test['interpretation']}")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    """Match Center: ingest match detail, or read one match back."""
    if args.action in ("ingest", "backfill", "live"):
        from .pipeline.ingest_match_detail import (
            backfill_match_detail,
            ingest_match_detail,
            refresh_live_matches,
        )

        if args.action == "live":
            report = refresh_live_matches(window_hours=args.hours, league_ids=args.league)
        elif args.game:
            report = ingest_match_detail(args.game, force=args.force)
        else:
            report = backfill_match_detail(args.league or ["ENG_PL"], seasons=args.season,
                                           limit=args.limit, refresh=args.force)
        _print(report.to_dict())
        return 0

    if not args.game:
        print("a game uid is required: divinelines match show <game_uid>")
        return 2
    game_uid = args.game[0]

    from .matchcenter.report import match_report
    from .matchcenter.service import MatchNotFound, match_center

    try:
        if args.action == "report":
            _print(match_report(game_uid, minute=args.minute))
            return 0
        payload = match_center(game_uid, minute=args.minute)
    except MatchNotFound:
        print(f"unknown match '{game_uid}'")
        return 1

    if args.json:
        _print(payload)
        return 0

    match, state = payload["match"], payload["state"]
    print("")
    print(f"  {match['home']['name']} {match['home']['score']}"
          f"-{match['away']['score']} {match['away']['name']}"
          f"   [{state['state']} / {state['mode']}]")
    print(f"  {match.get('league_name') or match['league_id']} {match['season']}"
          f"  {match['game_date']}  {match.get('venue') or 'venue unknown'}")

    quality = payload["quality"]
    print("")
    print(f"  data quality: {quality['grade']}")
    for component in quality["components"]:
        mark = {"present": "ok  ", "partial": "part", "absent": "--  "}[component["state"]]
        print(f"    [{mark}] {component['label']:<20} {component['detail']}")

    momentum = payload["momentum"].get("summary") or {}
    if momentum.get("available"):
        print("")
        print(f"  momentum ({momentum['version']}): "
              f"{momentum['minutes_home_ahead']} min home / "
              f"{momentum['minutes_away_ahead']} min away")

    shots = payload["shots"]
    print(f"  shots: {shots['total_shots']} recorded, {shots['located']} with a position")

    for event in payload["events"]:
        if event["event_type"] in ("goal", "penalty_scored", "own_goal", "red_card"):
            print(f"    {event['clock_display'] or '?':>7}  {event['event_type']:<15}"
                  f" {event.get('player_name') or ''} ({event.get('team_name') or '?'})")
    print()
    return 0


def cmd_lineups(args: argparse.Namespace) -> int:
    from .pipeline.ingest_lineups import backfill_historical_lineups, ingest_upcoming_lineups

    if args.historical:
        report = backfill_historical_lineups(args.league or ["ENG_PL"], limit=args.limit,
                                             seasons=args.season)
    else:
        report = ingest_upcoming_lineups(hours_ahead=args.hours, league_ids=args.league)
    _print(report.to_dict())
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from .pipeline.backfill_odds import backfill_nba_odds, backfill_soccer_espn_ids

    if args.what == "nba-odds":
        _print(backfill_nba_odds(seasons=args.season, limit_days=args.limit_days,
                                 force=args.force).to_dict())
    else:
        _print(backfill_soccer_espn_ids(args.league or ["ENG_PL"], days_back=args.days_back))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .pipeline.replay import replay_nba

    _print(replay_nba(seasons=args.season, price_at=args.price_at).to_dict())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .betting.ledger import performance_summary

    for dimension in (None, "sport", "edge_bucket", "odds_bucket"):
        frame = performance_summary(dimension, args.mode)
        print(f"\n--- {dimension or 'overall'} ---")
        _print(frame)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .analytics.model_health import compute_health
    from .betting.closing_line import closing_line_coverage
    from .betting.settlement import settlement_state
    from .data.freshness import freshness_report
    from .db.connection import query_df
    from .db.validation import run_database_health_checks
    from .models.registry import list_models

    def section(title: str) -> None:
        print()
        print(f"=== {title} ===")

    section("data")
    _print(query_df(
        "SELECT sport, COUNT(*) AS games, MIN(game_date) AS first, MAX(game_date) AS last "
        "FROM games GROUP BY sport"
    ))

    # Aggregated by source. Per-event adapters report health per feed, so a
    # status screen that listed every fetched fixture would be unreadable.
    section("sources")
    _print(query_df(
        """
        SELECT source, COUNT(*) AS datasets,
               SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok,
               SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
               SUM(CASE WHEN status = 'degraded' THEN 1 ELSE 0 END) AS degraded,
               MAX(last_success) AS last_success
        FROM source_status GROUP BY source ORDER BY source
        """
    ))

    section("freshness")
    _print(pd.DataFrame([
        {"dataset": f.dataset, "kind": f.kind, "state": f.state,
         "age_minutes": f.age_minutes, "usable": f.is_usable}
        for f in freshness_report()
    ]))

    section("closing lines")
    _print(closing_line_coverage())
    print("  settlement:", settlement_state())

    section("lineups")
    lineups = query_df(
        "SELECT sport, lineup_state, COUNT(DISTINCT game_uid) AS games, "
        "MAX(observed_at) AS latest FROM lineup_observations GROUP BY sport, lineup_state"
    )
    _print(lineups if not lineups.empty else "(no lineup observations)")

    section("models")
    models = list_models(limit=5)
    _print(models[["model_id", "sport", "kind", "trained_at"]] if not models.empty else models)

    section("model status")
    for sport in ("nba", "soccer"):
        health = compute_health(sport)
        print(f"  {sport:7s} {health.status:26s} n={health.sample_size}")
        print(f"          {health.status_reason}")

    section("validation")
    report = run_database_health_checks()
    _print({"ok": report.ok,
            "issues": [{"severity": i.severity, "code": i.code, "detail": i.detail}
                       for i in report.issues]})
    return 0 if report.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - process entry
    from .api.app import run

    run(host=args.host, port=args.port, reload=args.reload)
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="divinelines", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default=settings.log_level)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate", help="import the legacy database")
    migrate.add_argument("--lenient", action="store_true",
                         help="write data even if validation reports critical issues")
    migrate.set_defaults(func=cmd_migrate)

    refresh = subparsers.add_parser("refresh", help="fetch fresh data")
    refresh.add_argument("--sport", nargs="+", default=["nba", "soccer"],
                         choices=["nba", "soccer"])
    refresh.add_argument("--league", nargs="+", default=None)
    refresh.add_argument("--no-odds", action="store_true", help="skip the odds API")
    refresh.add_argument("--no-match-detail", action="store_true",
                         help="skip re-reading events for matches in progress")
    refresh.set_defaults(func=cmd_refresh)

    train = subparsers.add_parser("train", help="train and register models")
    train.add_argument("--sport", nargs="+", default=["nba", "soccer"],
                       choices=["nba", "soccer"])
    train.add_argument("--variant", default=None,
                       help="feature variant; defaults to the ablation-selected set")
    train.add_argument("--use-market", action="store_true",
                       help="include the market as a soccer ensemble component")
    train.set_defaults(func=cmd_train)

    scan_parser = subparsers.add_parser("scan", help="generate predictions and +EV opportunities")
    scan_parser.add_argument("--sport", nargs="+", default=["nba", "soccer"],
                             choices=["nba", "soccer"])
    scan_parser.add_argument("--days", type=int, default=3)
    scan_parser.add_argument("--limit", type=int, default=20)
    scan_parser.add_argument("--paper", action="store_true", help="open paper bets")
    scan_parser.add_argument("--dry-run", action="store_true",
                             help="show the slate without writing predictions")
    scan_parser.set_defaults(func=cmd_scan)

    backtest = subparsers.add_parser("backtest", help="walk-forward evaluation")
    backtest.add_argument("--sport", nargs="+", default=["nba", "soccer"],
                          choices=["nba", "soccer"])
    backtest.add_argument("--seasons", nargs="+", default=None)
    backtest.add_argument("--league", nargs="+", default=None)
    backtest.add_argument("--variant", default="full")
    backtest.add_argument("--use-market", action="store_true")
    backtest.add_argument("--price-book", default=None,
                          help="book to strike bets at (e.g. market_best)")
    backtest.add_argument("--save", action="store_true", help="write summaries to artifacts")
    backtest.set_defaults(func=cmd_backtest)

    ablate = subparsers.add_parser("ablate", help="feature ablation study")
    ablate.add_argument("--sport", nargs=1, default=["nba"], choices=["nba", "soccer"])
    ablate.add_argument("--seasons", nargs="+", default=None)
    ablate.set_defaults(func=cmd_ablate)

    settle = subparsers.add_parser("settle", help="resolve closes, compute CLV, grade results")
    settle.add_argument("--sport", default=None, choices=["nba", "soccer"])
    settle.add_argument("--date", default=None, help="only events on this date (YYYY-MM-DD)")
    settle.add_argument("--dry-run", action="store_true",
                        help="report what would be settled without writing")
    settle.add_argument("--full", action="store_true",
                        help="full reconciliation instead of incremental")
    settle.add_argument("--cutoff", type=int, default=0,
                        help="seconds before event start after which prices are ignored")
    settle.add_argument("--close-aggregation", default="median",
                        choices=["median", "best", "consensus", "book"])
    settle.add_argument("--close-book", default=None,
                        help="bookmaker to use when --close-aggregation=book")
    settle.set_defaults(func=cmd_settle)

    health = subparsers.add_parser("health", help="model health")
    health.add_argument("--sport", nargs="+", default=["nba", "soccer"],
                        choices=["nba", "soccer"])
    health.add_argument("--window", nargs="+", default=None,
                        help="all_time, last_90d, last_30d")
    health.add_argument("--market", default=None)
    health.add_argument("--persist", action="store_true", help="store a health snapshot")
    health.set_defaults(func=cmd_health)

    clv = subparsers.add_parser("clv", help="closing line value")
    clv.add_argument("--sport", default=None, choices=["nba", "soccer"])
    clv.add_argument("--basis", default="consensus", choices=["consensus", "same_book"])
    clv.add_argument("--cohort", default=None,
                     help="sport, market, time_to_event, edge_bucket, model_version, ...")
    clv.add_argument("--skill", action="store_true",
                     help="decompose CLV into shopping, drift and model selection")
    clv.add_argument("--mode", default="backtest", help="ledger mode for the skill test")
    clv.set_defaults(func=cmd_clv)

    lineups = subparsers.add_parser("lineups", help="ingest lineups")
    lineups.add_argument("--league", nargs="+", default=None)
    lineups.add_argument("--hours", type=int, default=6,
                         help="look this far ahead for upcoming fixtures")
    lineups.add_argument("--historical", action="store_true",
                         help="backfill actual XIs for finished matches (research only)")
    lineups.add_argument("--season", nargs="+", default=None)
    lineups.add_argument("--limit", type=int, default=400)
    lineups.set_defaults(func=cmd_lineups)

    match = subparsers.add_parser("match", help="soccer match centre: ingest or read a match")
    match.add_argument("action", choices=["ingest", "backfill", "live", "show", "report"])
    match.add_argument("game", nargs="*", default=None, help="game uid(s)")
    match.add_argument("--league", nargs="+", default=None)
    match.add_argument("--season", nargs="+", default=None)
    match.add_argument("--limit", type=int, default=400)
    match.add_argument("--hours", type=int, default=3,
                       help="how far back to look for a match in progress")
    match.add_argument("--minute", type=float, default=None,
                       help="replay position: show the match as it stood at this minute")
    match.add_argument("--force", action="store_true", help="re-read even if already stored")
    match.add_argument("--json", action="store_true", help="raw payload instead of a summary")
    match.set_defaults(func=cmd_match)

    backfill = subparsers.add_parser("backfill", help="historical odds and id mapping")
    backfill.add_argument("what", choices=["nba-odds", "soccer-ids"])
    backfill.add_argument("--season", nargs="+", default=None)
    backfill.add_argument("--league", nargs="+", default=None)
    backfill.add_argument("--limit-days", type=int, default=None)
    backfill.add_argument("--days-back", type=int, default=400)
    backfill.add_argument("--force", action="store_true")
    backfill.set_defaults(func=cmd_backfill)

    replay = subparsers.add_parser("replay", help="replay seasons into the ledger")
    replay.add_argument("--season", nargs="+", default=None)
    replay.add_argument("--price-at", default="best", choices=["best", "consensus"])
    replay.set_defaults(func=cmd_replay)

    evaluate = subparsers.add_parser("evaluate", help="realised performance")
    evaluate.add_argument("--mode", default=None, choices=["paper", "live", None])
    evaluate.set_defaults(func=cmd_evaluate)

    status = subparsers.add_parser("status", help="system health")
    status.set_defaults(func=cmd_status)

    serve = subparsers.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, force=True)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted")
        return 130
    except Exception as exc:
        log.error("command failed", extra={"command": args.command, "error": str(exc)})
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
