"""Train the spread-convergence booster from live API history."""

from __future__ import annotations

from zipadvisors.model import train_model


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "n/a"


def main() -> None:
    print("Fetching Kalshi / Polymarket history for focus markets...")
    stats = train_model()
    print(
        f"{stats['target']}: {stats['rows']} rows, "
        f"val accuracy={_fmt(stats.get('val_accuracy'))}, "
        f"val AUC={_fmt(stats.get('val_auc'))}, "
        f"base rate={_fmt(stats.get('base_rate'))}, "
        f"best_iteration={stats.get('best_iteration')}"
    )
    print("Wrote models/xgb_convergence.json")


if __name__ == "__main__":
    main()
