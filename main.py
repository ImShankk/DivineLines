"""DivineLines entry point.

The v1 interactive menu lived here.  It has been replaced by a proper CLI
(``divinelines.cli``) so that every operation is scriptable, testable and
usable from cron — but this file stays as the familiar way in, and running it
with no arguments still opens an interactive menu.

    python main.py                 interactive menu
    python main.py status          any CLI command works directly
    python main.py scan --sport soccer

The v1 scripts under ``core/``, ``nba/`` and ``soccer/`` are untouched and
still run; see the "Where the v1 code went" section of the README.
"""

from __future__ import annotations

import sys

from divinelines.cli import main as cli_main

MENU = """
=====================================================
                  DIVINELINES 2.0
=====================================================
 [1] Status          data, sources, models, validation
 [2] Refresh         fetch results, fixtures, injuries, odds
 [3] Train           fit and register models
 [4] Scan            generate predictions and +EV opportunities
 [5] Backtest        walk-forward evaluation
 [6] Serve           run the API on port 8000
 [7] Exit
=====================================================
"""

CHOICES = {
    "1": ["status"],
    "2": ["refresh"],
    "3": ["train"],
    "4": ["scan"],
    "5": ["backtest", "--save"],
    "6": ["serve"],
}


def interactive() -> int:
    while True:
        print(MENU)
        choice = input("Select a command (1-7): ").strip()
        if choice == "7":
            print("Shutting down DivineLines.")
            return 0
        arguments = CHOICES.get(choice)
        if not arguments:
            print("\n[!] Invalid selection.\n")
            continue
        try:
            cli_main(arguments)
        except SystemExit as exit_signal:  # a subcommand asked to stop
            if exit_signal.code:
                return int(exit_signal.code)
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]) if len(sys.argv) > 1 else interactive())
