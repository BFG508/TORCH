"""Run independent analytic mesh studies and save a JSON report."""

import argparse
import json
from pathlib import Path

from cfdSolver.torchVerification import verification_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Two smaller meshes per case, suitable for CI",
    )
    parser.add_argument("--out", default="out/verification.json", type=Path)
    args = parser.parse_args()
    report = verification_report(args.quick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        orders = ", ".join(
            "roundoff" if x is None else f"{x:.2f}"
            for x in check["observed_density_orders"]
        )
        print(
            f"{status} {check['kernel']} {check['dimension']}D "
            f"{check['case']}: density orders {orders}"
        )
    print(f"Report: {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
