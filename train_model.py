"""Train XGBoost direction models from live allowlisted ticks."""

from __future__ import annotations

from db import ensure_schema, tick_count
from model import train_model


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "n/a"


def main() -> None:
    ensure_schema()
    n = tick_count()
    if n == 0:
        raise SystemExit("No ticks in the database. Start the Kalshi and Polymarket scrapers first.")
    print(f"Training on {n} ticks")
    metrics = train_model()
    for target, stats in metrics.items():
        print(
            f"{target}: {stats['rows']} rows, "
            f"val accuracy={_fmt(stats.get('val_accuracy'))}, "
            f"val AUC={_fmt(stats.get('val_auc'))}, "
            f"best_iteration={stats.get('best_iteration')}"
        )


if __name__ == "__main__":
    main()
