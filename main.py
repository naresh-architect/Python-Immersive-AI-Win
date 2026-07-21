"""
SupportPilot — CLI entry point.

Usage:
    python3 main.py --init-db
    python3 main.py --test-suite
    python3 main.py --ticket "Where is my order?" --customer CUST001
"""
from __future__ import annotations

import argparse
import json

import database
from pipeline import run_ticket, pretty_print_report


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportPilot — ShopStream India support automation")
    parser.add_argument("--init-db", action="store_true", help="(Re)seed the mock database")
    parser.add_argument("--test-suite", action="store_true", help="Run the 8-scenario test suite")
    parser.add_argument("--ticket", type=str, help="Free-text ticket content to run through the pipeline")
    parser.add_argument("--ticket-id", type=str, default="T-CLI-001")
    parser.add_argument("--customer", type=str, default=None, help="Customer ID, e.g. CUST001")
    parser.add_argument("--json", action="store_true", help="Print the full case report as JSON")
    args = parser.parse_args()

    if args.init_db:
        database.init_db(force=True)
        print(f"Database seeded at {database.DB_PATH}")
        return

    if args.test_suite:
        from test_tickets import run_suite
        run_suite()
        return

    if args.ticket:
        report = run_ticket(args.ticket_id, args.ticket, customer_id=args.customer)
        pretty_print_report(report)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
