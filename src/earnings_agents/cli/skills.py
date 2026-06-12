"""``uv run earnings-skills`` — list the analysis failure-mode skill catalog.

Prints every entry in ``analysis/skills.SKILL_REGISTRY`` so the available
failure-mode detectors and their curated remediations are discoverable without
reading source. Useful when triaging a degraded run or deciding whether a new
checker is needed.

Usage::

    uv run earnings-skills                 # summary table
    uv run earnings-skills --verbose       # include full remediation text
    uv run earnings-skills --id gross_profit_identity   # detail for one skill
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from earnings_agents.analysis.skills import SKILL_REGISTRY, skill_by_id

console = Console()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="earnings-skills",
        description="List the analysis failure-mode skill catalog.",
    )
    parser.add_argument(
        "--id",
        dest="skill_id",
        default=None,
        help="Show full detail for a single skill by its id.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include full remediation text in the table.",
    )
    return parser.parse_args(argv)


def _print_one(skill_id: str) -> None:
    skill = skill_by_id(skill_id)
    if skill is None:
        ids = ", ".join(s.id for s in SKILL_REGISTRY)
        console.print(f"[red]No skill with id '{skill_id}'.[/red]\nKnown ids: {ids}")
        sys.exit(1)

    console.print(f"[bold cyan]{skill.id}[/bold cyan] — {skill.title}")
    console.print(f"[dim]finding types:[/dim] {', '.join(skill.finding_types)}")
    console.print(
        f"[dim]detector:[/dim] "
        f"{'(special-invocation, no registry detector)' if skill.detector is None else skill.detector.__name__}"
    )
    console.print("\n[bold]Remediation[/bold]")
    console.print(skill.remediation)
    if skill.notes:
        console.print("\n[bold]Notes[/bold]")
        console.print(skill.notes)


def _print_table(verbose: bool) -> None:
    table = Table(title=f"Failure-mode skill catalog ({len(SKILL_REGISTRY)} skills)")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("title")
    table.add_column("finding types", style="magenta")
    table.add_column("detector", style="green")
    if verbose:
        table.add_column("remediation")

    for skill in SKILL_REGISTRY:
        detector = "—" if skill.detector is None else skill.detector.__name__
        row = [
            skill.id,
            skill.title,
            ", ".join(skill.finding_types),
            detector,
        ]
        if verbose:
            row.append(skill.remediation)
        table.add_row(*row)

    console.print(table)
    detectors = sum(1 for s in SKILL_REGISTRY if s.detector is not None)
    console.print(
        f"[dim]{detectors} run as registry detectors; "
        f"{len(SKILL_REGISTRY) - detectors} are special-invocation (presence, source grounding).[/dim]"
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.skill_id:
        _print_one(args.skill_id)
        return
    _print_table(args.verbose)


if __name__ == "__main__":
    main()
