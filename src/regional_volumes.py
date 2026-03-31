from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOCUS_STRUCTURES = [
    "Hippocampus",
    "Amygdala",
    "Thalamus-Proper",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Accumbens-Area",
    "Ventral-DC",
    "Lateral-Ventricle",
    "Inf-Lat-Vent",
]

DISPLAY_NAMES = {
    "Hippocampus": "Hippocampus",
    "Amygdala": "Amygdala",
    "Thalamus-Proper": "Thalamus",
    "Caudate": "Caudate",
    "Putamen": "Putamen",
    "Pallidum": "Pallidum",
    "Accumbens-Area": "Accumbens",
    "Ventral-DC": "Ventral DC",
    "Lateral-Ventricle": "Lateral Ventricle",
    "Inf-Lat-Vent": "Inferior Lateral Ventricle",
}


def resolve_project_root(cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    if (cwd / "data").exists() and (cwd / "outputs").exists():
        return cwd
    if (cwd.parent / "data").exists() and (cwd.parent / "outputs").exists():
        return cwd.parent
    raise FileNotFoundError(f"Could not locate project root from {cwd}")


def load_label_lookup(metadata_path: Path) -> dict[int, str]:
    metadata = json.loads(metadata_path.read_text())
    return {
        int(label_id): label_name
        for label_id, label_name in metadata["network_data_format"]["outputs"]["pred"]["channel_def"].items()
    }


def split_label_name(label_name: str) -> tuple[str, str]:
    if label_name.startswith("Left-"):
        return "Left", label_name[len("Left-") :]
    if label_name.startswith("Right-"):
        return "Right", label_name[len("Right-") :]
    return "Midline", label_name


def load_tiv_cm3(tissue_volume_csv: Path) -> float | None:
    if not tissue_volume_csv.exists():
        return None
    df = pd.read_csv(tissue_volume_csv)
    tiv_row = df[df["tissue"] == "TIV"]
    if tiv_row.empty:
        return None
    return float(tiv_row.iloc[0]["volume_cm3"])


def compute_label_volume_table(
    label_map: np.ndarray,
    label_lookup: dict[int, str],
    voxel_volume_mm3: float,
    tiv_cm3: float | None = None,
) -> pd.DataFrame:
    unique_ids, counts = np.unique(label_map.astype(np.uint16), return_counts=True)
    count_map = {int(label_id): int(count) for label_id, count in zip(unique_ids, counts)}

    rows: list[dict[str, float | int | str]] = []
    for label_id in sorted(label_lookup):
        if label_id == 0:
            continue
        label_name = label_lookup[label_id]
        side, region_name = split_label_name(label_name)
        voxel_count = count_map.get(label_id, 0)
        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_cm3 = volume_mm3 / 1000.0
        row: dict[str, float | int | str] = {
            "label_id": label_id,
            "label_name": label_name,
            "side": side,
            "region_name": region_name,
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
            "volume_cm3": volume_cm3,
        }
        if tiv_cm3 is not None and tiv_cm3 > 0:
            row["fraction_of_tiv"] = volume_cm3 / tiv_cm3
            row["percent_of_tiv"] = 100.0 * volume_cm3 / tiv_cm3
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(["volume_cm3", "label_id"], ascending=[False, True]).reset_index(drop=True)
    return table


def compute_bilateral_summary(label_table: pd.DataFrame, tiv_cm3: float | None = None) -> pd.DataFrame:
    paired = label_table[label_table["side"].isin(["Left", "Right"])].copy()
    if paired.empty:
        return pd.DataFrame()

    rows = []
    for region_name, region_df in paired.groupby("region_name", sort=True):
        left_df = region_df[region_df["side"] == "Left"]
        right_df = region_df[region_df["side"] == "Right"]

        left_voxels = int(left_df["voxel_count"].sum())
        right_voxels = int(right_df["voxel_count"].sum())
        left_volume_cm3 = float(left_df["volume_cm3"].sum())
        right_volume_cm3 = float(right_df["volume_cm3"].sum())
        total_volume_cm3 = left_volume_cm3 + right_volume_cm3

        row = {
            "region_name": region_name,
            "left_label_ids": left_df["label_id"].astype(int).tolist(),
            "right_label_ids": right_df["label_id"].astype(int).tolist(),
            "left_voxel_count": left_voxels,
            "right_voxel_count": right_voxels,
            "left_volume_cm3": left_volume_cm3,
            "right_volume_cm3": right_volume_cm3,
            "bilateral_volume_cm3": total_volume_cm3,
            "asymmetry_index": np.nan if total_volume_cm3 == 0 else (right_volume_cm3 - left_volume_cm3) / total_volume_cm3,
        }
        row["asymmetry_percent"] = np.nan if total_volume_cm3 == 0 else 100.0 * row["asymmetry_index"]
        if tiv_cm3 is not None and tiv_cm3 > 0:
            row["bilateral_fraction_of_tiv"] = total_volume_cm3 / tiv_cm3
            row["bilateral_percent_of_tiv"] = 100.0 * total_volume_cm3 / tiv_cm3
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("bilateral_volume_cm3", ascending=False).reset_index(drop=True)
    return summary


def compute_midline_summary(label_table: pd.DataFrame, tiv_cm3: float | None = None) -> pd.DataFrame:
    midline = label_table[label_table["side"] == "Midline"].copy()
    if midline.empty:
        return pd.DataFrame()
    summary = midline[["label_id", "label_name", "region_name", "voxel_count", "volume_mm3", "volume_cm3"]].copy()
    if tiv_cm3 is not None and tiv_cm3 > 0:
        summary["fraction_of_tiv"] = summary["volume_cm3"] / tiv_cm3
        summary["percent_of_tiv"] = 100.0 * summary["volume_cm3"] / tiv_cm3
    return summary.sort_values("volume_cm3", ascending=False).reset_index(drop=True)


def build_focus_table(
    bilateral_summary: pd.DataFrame,
    structures: Iterable[str] = FOCUS_STRUCTURES,
) -> pd.DataFrame:
    focus = bilateral_summary[bilateral_summary["region_name"].isin(list(structures))].copy()
    if focus.empty:
        return focus
    focus["display_name"] = focus["region_name"].map(DISPLAY_NAMES).fillna(focus["region_name"])
    order = {name: idx for idx, name in enumerate(structures)}
    focus["sort_key"] = focus["region_name"].map(order)
    focus = focus.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return focus


def save_table(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)


def plot_focus_summary(focus_table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if focus_table.empty:
        raise ValueError("Focus table is empty; nothing to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    y = np.arange(len(focus_table))

    axes[0].barh(y - 0.18, focus_table["left_volume_cm3"], height=0.35, color="#4c78a8", label="Left")
    axes[0].barh(y + 0.18, focus_table["right_volume_cm3"], height=0.35, color="#f58518", label="Right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(focus_table["display_name"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Volume (cm^3)")
    axes[0].set_title("Left vs right subcortical volumes")
    axes[0].legend(frameon=False)

    axes[1].barh(y, focus_table["bilateral_volume_cm3"], color="#54a24b")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(focus_table["display_name"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Volume (cm^3)")
    axes[1].set_title("Bilateral totals and asymmetry")

    max_total = float(focus_table["bilateral_volume_cm3"].max())
    for idx, (_, row) in enumerate(focus_table.iterrows()):
        axes[1].text(
            row["bilateral_volume_cm3"] + max_total * 0.01,
            idx,
            f"{row['asymmetry_percent']:+.1f}%",
            va="center",
            fontsize=10,
        )

    fig.suptitle("Advanced parcellation regional-volume summary", fontsize=15)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
