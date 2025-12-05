#!/usr/bin/env python3
"""
LHCb-style tracking plots for VELO toy model results.

This module **replaces** the previous plotting utility. It consumes a DataFrame with
exactly the columns you listed and produces the four requested plots (9 points each),
plus a bonus scatter. Aggregation is mean over rows with RMS error bars (≈ your 10 events).

Required columns (verbatim):
Index([
  'file', 'p_hit_res', 'p_multi_scatter', 'p_ghost_rate', 'p_drop_rate',
  'p_repeat', 'p_epsilon', 'p_layers', 'p_dz_mm', 'p_seed',
  'm_total_truth_tracks', 'm_total_reconstructible_truth',
  'm_total_rec_candidates', 'm_candidate_rec_ids', 'm_ghost_rec_ids',
  'm_primary_matches', 'm_truth_to_recs', 'm_reconstruction_efficiency',
  'm_ghost_rate', 'm_clone_fraction_total', 'm_clone_fraction_among_matched',
  'm_purity_all_matched', 'm_purity_primary_only', 'm_hit_efficiency_mean',
  'm_hit_efficiency_weighted', 'm_n_ghosts', 'm_n_clones',
  'm_n_matched_reco', 'm_n_matched_truth'
])

Usage (as a script):
  python lhcb_tracking_plots.py --csv results.csv --out perf --ms_fixed 0.0002

As a library:
  from lhcb_tracking_plots import plot_all
  out = plot_all(df, out_prefix="perf", ms_fixed=2e-4)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================= helpers =============================

# -------- Schema normalisation (handles m_m_* -> m_* prefixes) --------

def normalize_df_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical 'm_*' columns.
    Your current CSV seems to use 'm_m_*' names; we mirror them to 'm_*'.
    """
    df = df.copy()
    # Map of metric stems we care about
    stems = [
        'total_truth_tracks', 'total_reconstructible_truth', 'total_rec_candidates',
        'candidate_rec_ids', 'ghost_rec_ids', 'primary_matches', 'truth_to_recs',
        'reconstruction_efficiency', 'ghost_rate', 'clone_fraction_total',
        'clone_fraction_among_matched', 'purity_all_matched', 'purity_primary_only',
        'hit_efficiency_mean', 'hit_efficiency_weighted', 'n_ghosts', 'n_clones',
        'n_matched_reco', 'n_matched_truth'
    ]
    for stem in stems:
        mm = f'm_m_{stem}'
        m = f'm_{stem}'
        if m not in df.columns and mm in df.columns:
            df[m] = df[mm]
    return df

# Thresholded plotting from per-track tables ---------------------------------
# Assumes a directory with per-event track tables named like '*.parquet' or '*.csv'.

def _load_track_tables(tracks_dir: Path) -> pd.DataFrame:
    files = sorted(list(tracks_dir.glob('*.parquet'))) or sorted(list(tracks_dir.glob('*.csv')))
    if not files:
        raise FileNotFoundError(f"No per-track tables found under {tracks_dir} (expected .parquet or .csv)")
    dfs = []
    for f in files:
        if f.suffix == '.parquet':
            dfs.append(pd.read_parquet(f))
        else:
            dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    return df


def _merge_params_with_tracks(results_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
    # Expect both to have an event/file key. Use 'file' if present, else create a row index.
    if 'file' in results_df and 'file' in track_df:
        return track_df.merge(results_df[['file','p_hit_res','p_multi_scatter','p_drop_rate','p_epsilon']], on='file', how='left')
    # fallback: attach first-row params to all tracks per chunk (less ideal)
    track_df = track_df.copy()
    for col in ['p_hit_res','p_multi_scatter','p_drop_rate','p_epsilon']:
        if col not in track_df and col in results_df:
            track_df[col] = results_df[col].iloc[0]
    return track_df


def _threshold_accept_mask(trk: pd.DataFrame, purity_min: float, completeness_min: Optional[float], min_shared_hits: int) -> pd.Series:
    mask = (trk['purity'] >= float(purity_min)) & (trk['correct_hits'] >= int(min_shared_hits))
    if completeness_min is not None and 'completeness' in trk:
        mask &= (trk['completeness'] >= float(completeness_min))
    return mask


def _aggregate_thresholded(trk: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # candidate reco only
    cand = trk[trk['candidate'] == True].copy()  # noqa: E712
    # counts
    g = cand.groupby(group_cols, dropna=False)
    df_counts = g.agg(total_rec_candidates=('rec_id','count')).reset_index()
    # ghosts (not accepted)
    df_counts['n_ghosts'] = (g.apply(lambda x: int((~x['accepted']).sum()))).values
    # simple rates per group
    df_counts['ghost_rate'] = df_counts['n_ghosts'] / df_counts['total_rec_candidates']
    return df_counts


# Thresholded plotting from per-track tables ---------------------------------
# Assumes a directory with per-event tables named like '*.parquet' or '*.csv'.

def _load_track_tables(tracks_dir: Path) -> pd.DataFrame:
    files = sorted(list(tracks_dir.glob('*.parquet'))) or sorted(list(tracks_dir.glob('*.csv')))
    if not files:
        raise FileNotFoundError(f"No per-track tables found under {tracks_dir} (expected .parquet or .csv)")
    dfs = []
    for f in files:
        if f.suffix == '.parquet':
            dfs.append(pd.read_parquet(f))
        else:
            dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    return df


def _merge_params_with_tracks(results_df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
    # Expect both to have an event/file key. Use 'file' if present, else create a row index.
    if 'file' in results_df and 'file' in track_df:
        return track_df.merge(results_df[['file','p_hit_res','p_multi_scatter','p_drop_rate','p_epsilon']], on='file', how='left')
    # fallback: attach first-row params to all tracks per chunk (less ideal)
    track_df = track_df.copy()
    for col in ['p_hit_res','p_multi_scatter','p_drop_rate','p_epsilon']:
        if col not in track_df and col in results_df:
            track_df[col] = results_df[col].iloc[0]
    return track_df


def _threshold_accept_mask(trk: pd.DataFrame, purity_min: float, completeness_min: Optional[float], min_shared_hits: int) -> pd.Series:
    mask = (trk['purity'] >= float(purity_min)) & (trk['correct_hits'] >= int(min_shared_hits))
    if completeness_min is not None and 'completeness' in trk:
        mask &= (trk['completeness'] >= float(completeness_min))
    return mask


def _aggregate_thresholded(trk: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # candidate reco only
    cand = trk[trk['candidate'] == True].copy()  # noqa: E712
    # counts
    g = cand.groupby(group_cols, dropna=False)
    df_counts = g.agg(total_rec_candidates=('rec_id','count')).reset_index()
    # ghosts (not accepted)
    df_counts['n_ghosts'] = (g.apply(lambda x: int((~x['accepted']).sum()))).values
    # simple rates per group
    df_counts['ghost_rate'] = df_counts['n_ghosts'] / df_counts['total_rec_candidates']
    return df_counts


def generate_thresholded_plots(results_df: pd.DataFrame, tracks_dir: Path, purity_min: float = 0.7,
                               completeness_min: Optional[float] = None, min_shared_hits: int = 0,
                               out_prefix: str = 'perf_thr', ms_values: Sequence[float] | None = None,
                               drop_values: Sequence[float] | None = None, hit_res_values: Sequence[float] | None = None,
                               save_png: bool = True, save_csv: bool = True, drop_zero_tol: float = 1e-9):
    """Produce purity-thresholded curves using per-track tables generated by toy_validator.py.

    Expected per-track columns: ['rec_id','best_truth_id','accepted','candidate','rec_hits','truth_hits','correct_hits','purity','completeness']
    This function merges control parameters from the results CSV using the 'file' column when available.
    """
    trk = _load_track_tables(tracks_dir)
    if 'candidate' not in trk or 'purity' not in trk:
        raise ValueError('Per-track tables must include candidate and purity columns')
    trk = _merge_params_with_tracks(results_df, trk)

    # Acceptance by thresholds
    trk = trk.copy()
    trk['accepted'] = _threshold_accept_mask(trk, purity_min, completeness_min, min_shared_hits)

    # Baseline slice: drop≈0
    if 'p_drop_rate' in trk:
        vals = pd.to_numeric(trk['p_drop_rate'], errors='coerce').to_numpy(dtype=float)
        trk_base = trk[np.isclose(vals, 0.0, atol=drop_zero_tol)]
    else:
        trk_base = trk

    # -------- Thresholded Efficiency & Ghosts vs MS (baseline) --------
    # We can estimate ghost rate per group from candidate vs accepted counts. Efficiency needs reconstructible truth counts; as proxy, show ghost rate + accepted/candidate ratio.
    grp_cols_ms = [c for c in ['p_hit_res','p_multi_scatter'] if c in trk_base]
    thr_counts_ms = _aggregate_thresholded(trk_base, grp_cols_ms)
    if save_csv:
        thr_counts_ms.to_csv(f"{out_prefix}_thresholded_counts_vs_ms.csv", index=False)

    # Plot ghost rate and (1-ghost_rate) as accepted fraction
    if not thr_counts_ms.empty:
        x_ms = sorted(thr_counts_ms['p_multi_scatter'].dropna().unique()) if 'p_multi_scatter' in thr_counts_ms else None
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for label, sub in thr_counts_ms.groupby('p_hit_res'):
            sub = sub.sort_values('p_multi_scatter')
            axes[0].plot(sub['p_multi_scatter'], sub['ghost_rate'], marker='o', label=f'p_hit_res={label}')
            axes[1].plot(sub['p_multi_scatter'], 1.0 - sub['ghost_rate'], marker='o', label=f'p_hit_res={label}')
        axes[0].set_title(f'Thresholded Ghost Rate vs MS (purity≥{purity_min})'); axes[0].set_xlabel('MS'); axes[0].set_ylabel('Ghost Rate'); axes[0].grid(alpha=0.3)
        axes[1].set_title(f'Accepted/Candidate vs MS (≈1−ghost rate)'); axes[1].set_xlabel('MS'); axes[1].set_ylabel('Fraction'); axes[1].set_ylim(0,1.05); axes[1].grid(alpha=0.3)
        axes[0].legend(); axes[1].legend(); plt.tight_layout()
        if save_png:
            fig.savefig(f"{out_prefix}_thresholded_vs_ms.png", dpi=150, bbox_inches='tight')

    # -------- Thresholded Ghosts per event vs Hit Resolution (baseline) --------
    grp_cols_hr = [c for c in ['p_hit_res'] if c in trk_base]
    thr_counts_hr = _aggregate_thresholded(trk_base, grp_cols_hr)
    if save_csv:
        thr_counts_hr.to_csv(f"{out_prefix}_thresholded_counts_vs_hitres.csv", index=False)
    if not thr_counts_hr.empty:
        fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5))
        sub = thr_counts_hr.sort_values('p_hit_res')
        ax2.plot(sub['p_hit_res'], sub['ghost_rate'], marker='o')
        ax2.set_title(f'Thresholded Ghost Rate vs Hit Resolution (purity≥{purity_min})')
        ax2.set_xlabel('Hit Resolution'); ax2.set_ylabel('Ghost Rate'); ax2.grid(alpha=0.3)
        if save_png:
            fig2.savefig(f"{out_prefix}_thresholded_ghost_vs_hitres.png", dpi=150, bbox_inches='tight')

    print('[info] Thresholded plots generated from per-track tables.')

def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def rms(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    mu = float(arr.mean())
    return float(np.sqrt(np.mean((arr - mu) ** 2)))


def aggregate_mean_rms(df: pd.DataFrame, group_cols: Iterable[str], value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame(columns=list(group_cols) + ['mean', 'rms', 'n'])
    g = df.groupby(list(group_cols), dropna=False)[value_col]
    agg = g.agg(mean='mean', rms=lambda x: rms(x), n='count')
    return agg.reset_index()


def _slice_drop_zero(df: pd.DataFrame, drop_col: str = 'p_drop_rate', tol: float = 1e-9) -> tuple[pd.DataFrame, str]:
    if drop_col not in df:
        return df, 'all (no drop column)'
    vals = pd.to_numeric(df[drop_col], errors='coerce').to_numpy(dtype=float)
    mask = np.isclose(vals, 0.0, atol=tol)
    if mask.any():
        return df[mask], 'drop=0%'
    # fallback: pick per-(hit_res, multi_scatter) minimal drop rows
    group_cols = [c for c in ['p_hit_res', 'p_multi_scatter'] if c in df]
    if group_cols:
        min_per_group = df.groupby(group_cols)[drop_col].transform('min')
        return df[df[drop_col] == min_per_group], 'baseline≈min drop per group'
    # global min
    mind = df[drop_col].min()
    return df[df[drop_col] == mind], f'baseline≈min drop ({mind})'


def _x_order(values: Sequence[float] | None, discovered: Iterable[float]) -> list[float]:
    """Return a length-3 ordered list for the x-axis.
    If *values* is provided (explicit triplet), use it; else infer sorted unique from data.
    """
    if values is not None and len(values) > 0:
        return list(values)
    u = sorted(pd.Series(discovered).dropna().unique().tolist())
    return u


def _plot_errcurves(ax, agg_df: pd.DataFrame, x_col: str, label_col: str,
                    xlabel: str, ylabel: str, title: str,
                    x_values: Sequence[float] | None = None,
                    y_lims: Tuple[float, float] | None = None) -> None:
    if agg_df is None or agg_df.empty:
        ax.set_title(title + ' (no data)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.4)
        return
    xorder = _x_order(x_values, agg_df[x_col])
    for label, sub in agg_df.groupby(label_col):
        sub_sorted = sub.set_index(x_col).reindex(xorder).reset_index()
        ax.errorbar(sub_sorted[x_col], sub_sorted['mean'], yerr=sub_sorted['rms'],
                    marker='o', lw=1.5, capsize=4, label=f"{label_col}={label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_lims is not None:
        ax.set_ylim(*y_lims)
    ax.grid(alpha=0.4)
    ax.legend()

# ============================= public API =============================

# ---------- helpers for new analyses ----------

def _finite_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    # simple linear regression slope
    p = np.polyfit(x[mask], y[mask], deg=1)
    return float(p[0])


def _agg_curve(df: pd.DataFrame, by_cols: list[str], x_col: str, y_col: str) -> pd.DataFrame:
    return aggregate_mean_rms(df, by_cols + [x_col], y_col)


def _warn_purity_cut_posthoc(df: pd.DataFrame, threshold: float) -> None:
    # We can't post-hoc filter tracks by purity without per-track data.
    # This function prints a console note when called as a script.
    print(f"[note] Purity-thresholded plots assume your metrics were computed with purity_min={threshold}.   Post-hoc track-level purity filtering is not possible from the provided columns.")

def plot_all(df_in: pd.DataFrame, out_prefix: str = 'perf', ms_fixed: float = 2e-4,
             drop_zero_tol: float = 1e-9, save_png: bool = True, save_csv: bool = True,
             extra_scatter: bool = True,
             ms_values: Sequence[float] | None = None,
             drop_values: Sequence[float] | None = None,
             hit_res_values: Sequence[float] | None = None) -> dict:
    """Create the four requested plots (9 points each) and optional scatter.

    Parameters
    ----------
    df_in : DataFrame with the exact columns listed in the module docstring.
    out_prefix : filename prefix for outputs (CSV/PNG).
    ms_fixed : value of p_multi_scatter to hold fixed for drop-rate scans (default 0.0002).
    drop_zero_tol : tolerance for treating p_drop_rate as zero for baseline plots.
    save_png, save_csv : save figures/CSVs.
    extra_scatter : also generate Efficiency vs Ghost Rate scatter at baseline.
    ms_values, drop_values, hit_res_values : optional explicit x/value grids
      (pass three values each if you want to enforce the x-axis triplets and line set).
    """
    df = normalize_df_schema(df_in.copy())

    # Ensure numeric on fields we will plot
    num_cols = [
        'p_hit_res', 'p_multi_scatter', 'p_drop_rate',
        'm_reconstruction_efficiency', 'm_ghost_rate',
        'm_purity_primary_only', 'm_purity_all_matched',
        'm_hit_efficiency_weighted', 'm_hit_efficiency_mean',
        'm_clone_fraction_total'
    ]
    _to_numeric(df, num_cols)

    # Convert optional explicit sets to lists (used for ordering)
    if hit_res_values is None and 'p_hit_res' in df:
        hit_res_values = sorted(df['p_hit_res'].dropna().unique())
    if ms_values is None and 'p_multi_scatter' in df:
        ms_values = sorted(df['p_multi_scatter'].dropna().unique())
    if drop_values is None and 'p_drop_rate' in df:
        drop_values = sorted(df['p_drop_rate'].dropna().unique())

    # -------- 1) Track Efficiency vs Multiple Scattering (drop≈0%) --------
    base_ms, base_tag = _slice_drop_zero(df, 'p_drop_rate', tol=drop_zero_tol)
    eff_agg = aggregate_mean_rms(base_ms, ['p_hit_res', 'p_multi_scatter'], 'm_reconstruction_efficiency')

    fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 5))
    _plot_errcurves(
        ax1, eff_agg, x_col='p_multi_scatter', label_col='p_hit_res',
        xlabel='Multiple Scattering Parameter', ylabel='Track Efficiency',
        title=f'Track Efficiency vs Multiple Scattering ({base_tag})',
        x_values=ms_values, y_lims=(0.0, 1.05),
    )
    if save_png:
        fig1.savefig(f"{out_prefix}_eff_vs_ms.png", dpi=150, bbox_inches='tight')
    if save_csv:
        eff_agg.to_csv(f"{out_prefix}_eff_vs_ms.csv", index=False)

    # -------- 2) Ghost Rate vs Multiple Scattering (drop≈0%) --------
    ghost_agg = aggregate_mean_rms(base_ms, ['p_hit_res', 'p_multi_scatter'], 'm_ghost_rate')

    fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5))
    ymax_g = float(max(0.05, (ghost_agg['mean'].max() if not ghost_agg.empty else 0.05) * 1.25))
    _plot_errcurves(
        ax2, ghost_agg, x_col='p_multi_scatter', label_col='p_hit_res',
        xlabel='Multiple Scattering Parameter', ylabel='Ghost Rate',
        title=f'Ghost Rate vs Multiple Scattering ({base_tag})',
        x_values=ms_values, y_lims=(0.0, ymax_g),
    )
    if save_png:
        fig2.savefig(f"{out_prefix}_ghost_vs_ms.png", dpi=150, bbox_inches='tight')
    if save_csv:
        ghost_agg.to_csv(f"{out_prefix}_ghost_vs_ms.csv", index=False)

    # -------- 3) Hit Efficiency vs Drop Rate at fixed MS --------
    at_ms = df[np.isclose(df['p_multi_scatter'].to_numpy(dtype=float), float(ms_fixed), atol=1e-12)]
    he_col = 'm_hit_efficiency_weighted' if 'm_hit_efficiency_weighted' in at_ms.columns else ('m_hit_efficiency_mean' if 'm_hit_efficiency_mean' in at_ms.columns else None)
    if he_col is None:
        he_agg = pd.DataFrame(columns=['p_hit_res','p_drop_rate','mean','rms','n'])
    else:
        he_agg = aggregate_mean_rms(at_ms, ['p_hit_res', 'p_drop_rate'], he_col)

    fig3, ax3 = plt.subplots(1, 1, figsize=(7.5, 5))
    _plot_errcurves(
        ax3, he_agg, x_col='p_drop_rate', label_col='p_hit_res',
        xlabel='Hit Inefficiency (Drop Rate)', ylabel='Hit Efficiency (primary, |T|-weighted)',
        title=f'Hit Efficiency vs Drop Rate (MS={ms_fixed})',
        x_values=drop_values, y_lims=(0.0, 1.05),
    )
    if save_png:
        fig3.savefig(f"{out_prefix}_hit_eff_vs_drop_ms{ms_fixed}.png", dpi=150, bbox_inches='tight')
    if save_csv:
        he_agg.to_csv(f"{out_prefix}_hit_eff_vs_drop_ms{ms_fixed}.csv", index=False)

    # -------- 4) Hit Purity vs Drop Rate at fixed MS --------
    pur_col = 'm_purity_primary_only' if 'm_purity_primary_only' in at_ms.columns else ('m_purity_all_matched' if 'm_purity_all_matched' in at_ms.columns else None)
    if pur_col is None:
        pur_agg = pd.DataFrame(columns=['p_hit_res','p_drop_rate','mean','rms','n'])
    else:
        pur_agg = aggregate_mean_rms(at_ms, ['p_hit_res', 'p_drop_rate'], pur_col)

    fig4, ax4 = plt.subplots(1, 1, figsize=(7.5, 5))
    _plot_errcurves(
        ax4, pur_agg, x_col='p_drop_rate', label_col='p_hit_res',
        xlabel='Hit Inefficiency (Drop Rate)', ylabel='Track Purity (primary)',
        title=f'Hit Purity vs Drop Rate (MS={ms_fixed})',
        x_values=drop_values, y_lims=(0.0, 1.05),
    )
    if save_png:
        fig4.savefig(f"{out_prefix}_purity_vs_drop_ms{ms_fixed}.png", dpi=150, bbox_inches='tight')
    if save_csv:
        pur_agg.to_csv(f"{out_prefix}_purity_vs_drop_ms{ms_fixed}.csv", index=False)

    # -------- EXTRA: Efficiency vs Ghost rate scatter at baseline --------
    scatter_path = None
    if extra_scatter:
        eff_sc = eff_agg.rename(columns={'mean': 'eff_mean', 'rms': 'eff_rms'})
        ghost_sc = ghost_agg.rename(columns={'mean': 'ghost_mean', 'rms': 'ghost_rms'})
        key_cols = ['p_hit_res', 'p_multi_scatter']
        merged = pd.merge(eff_sc[key_cols + ['eff_mean', 'eff_rms']],
                          ghost_sc[key_cols + ['ghost_mean', 'ghost_rms']], on=key_cols, how='inner')
        fig5, ax5 = plt.subplots(1, 1, figsize=(6.5, 5.5))
        if not merged.empty:
            for hit_res, sub in merged.groupby('p_hit_res'):
                ax5.errorbar(sub['eff_mean'], sub['ghost_mean'],
                             xerr=sub['eff_rms'], yerr=sub['ghost_rms'],
                             fmt='o', label=f"p_hit_res={hit_res}")
        ax5.set_xlabel('Track Efficiency')
        ax5.set_ylabel('Ghost Rate')
        ax5.set_title(f'Efficiency vs Ghost Rate ({base_tag})')
        ax5.grid(alpha=0.4)
        ax5.legend()
        if save_png:
            scatter_path = f"{out_prefix}_eff_vs_ghost_scatter.png"
            fig5.savefig(scatter_path, dpi=150, bbox_inches='tight')
        if save_csv:
            merged.to_csv(f"{out_prefix}_eff_vs_ghost_scatter_table.csv", index=False)

    return {
        'efficiency_vs_ms': eff_agg,
        'ghost_vs_ms': ghost_agg,
        'hit_eff_vs_drop': he_agg,
        'purity_vs_drop': pur_agg,
        'baseline': base_tag,
        'scatter_table': scatter_path,
        'ms_values': list(ms_values) if ms_values is not None else None,
        'drop_values': list(drop_values) if drop_values is not None else None,
        'hit_res_values': list(hit_res_values) if hit_res_values is not None else None,
        'note': 'Includes helper functions for sensitivity analyses',
    }


def plot_all_extended(df_in: pd.DataFrame, out_prefix: str = 'perf_ext', ms_fixed: float = 2e-4,
                      drop_zero_tol: float = 1e-9, save_png: bool = True, save_csv: bool = True,
                      ms_values: Sequence[float] | None = None,
                      drop_values: Sequence[float] | None = None,
                      hit_res_values: Sequence[float] | None = None,
                      purity_min: float | None = None) -> dict:
    """Produce *all possible* plots with the available columns.

    Adds:
      - Clone fraction vs MS (baseline)
      - Efficiency vs Hit Resolution (baseline)
      - Event-level histograms: total_rec_candidates, n_ghosts, efficiency
      - Scatter diagnostics: clones vs total_truth, purity vs total_truth
      - Heatmaps (if grids exist): efficiency, ghost rate, purity over (MS, drop)
      - **NEW:** Grid of *all metrics vs Multiple Scattering* at baseline (drop≈0%)
      - **NEW:** Sensitivity plots (slopes) vs MS and vs Hit Resolution
      - **NEW:** Optional purity-thresholded repeats (requires upstream metrics computed with that cut)
      - **NEW:** Effect of epsilon window (plots vs p_epsilon)
    """
    df = normalize_df_schema(df_in.copy())

    # Ensure numeric
    num_cols = [
        'p_hit_res', 'p_multi_scatter', 'p_drop_rate',
        'm_reconstruction_efficiency', 'm_ghost_rate', 'm_clone_fraction_total', 'm_clone_fraction_among_matched',
        'm_purity_primary_only', 'm_purity_all_matched',
        'm_hit_efficiency_weighted', 'm_hit_efficiency_mean',
        'm_total_rec_candidates', 'm_n_ghosts', 'm_n_clones',
        'm_total_truth_tracks', 'm_total_reconstructible_truth'
    ]
    _to_numeric(df, num_cols)

    # Default x grids
    if hit_res_values is None and 'p_hit_res' in df:
        hit_res_values = sorted(df['p_hit_res'].dropna().unique())
    if ms_values is None and 'p_multi_scatter' in df:
        ms_values = sorted(df['p_multi_scatter'].dropna().unique())
    if drop_values is None and 'p_drop_rate' in df:
        drop_values = sorted(df['p_drop_rate'].dropna().unique())

    # Baseline slice (drop≈0)
    base_ms, base_tag = _slice_drop_zero(df, 'p_drop_rate', tol=drop_zero_tol)

    # ---- Clone fraction vs MS (baseline) ----
    clone_agg = aggregate_mean_rms(base_ms, ['p_hit_res', 'p_multi_scatter'], 'm_clone_fraction_total')
    fig_c, ax_c = plt.subplots(1, 1, figsize=(7.5, 5))
    ymax_c = float(max(0.05, (clone_agg['mean'].max() if not clone_agg.empty else 0.05) * 1.25))
    _plot_errcurves(
        ax_c, clone_agg, x_col='p_multi_scatter', label_col='p_hit_res',
        xlabel='Multiple Scattering Parameter', ylabel='Clone Fraction',
        title=f'Clone Fraction vs Multiple Scattering ({base_tag})',
        x_values=ms_values, y_lims=(0.0, ymax_c),
    )
    if save_png:
        fig_c.savefig(f"{out_prefix}_clone_vs_ms.png", dpi=150, bbox_inches='tight')
    if save_csv:
        clone_agg.to_csv(f"{out_prefix}_clone_vs_ms.csv", index=False)

    # ---- Efficiency vs Hit Resolution (baseline) ----
    eff_vs_res = aggregate_mean_rms(base_ms, ['p_hit_res'], 'm_reconstruction_efficiency')
    fig_ehr, ax_ehr = plt.subplots(1, 1, figsize=(6.5, 5))
    xorder_res = _x_order(hit_res_values, eff_vs_res['p_hit_res']) if not eff_vs_res.empty else hit_res_values
    if eff_vs_res is None or eff_vs_res.empty:
        ax_ehr.set_title('Track Efficiency vs Hit Resolution (no data)'); ax_ehr.axis('off')
    else:
        eff_vs_res_sorted = eff_vs_res.set_index('p_hit_res').reindex(xorder_res).reset_index()
        ax_ehr.errorbar(eff_vs_res_sorted['p_hit_res'], eff_vs_res_sorted['mean'], yerr=eff_vs_res_sorted['rms'],
                        marker='o', lw=1.5, capsize=4)
        ax_ehr.set_xlabel('Hit Resolution'); ax_ehr.set_ylabel('Track Efficiency');
        ax_ehr.set_title(f'Track Efficiency vs Hit Resolution ({base_tag})');
        ax_ehr.set_ylim(0.0, 1.05); ax_ehr.grid(alpha=0.4)
    if save_png:
        fig_ehr.savefig(f"{out_prefix}_eff_vs_hit_res.png", dpi=150, bbox_inches='tight')
    if save_csv and eff_vs_res is not None:
        eff_vs_res.to_csv(f"{out_prefix}_eff_vs_hit_res.csv", index=False)

    # ---- Event-level histograms ----
    fig_h, axes_h = plt.subplots(1, 3, figsize=(16, 4.5))
    # 1) total reconstructed (candidates)
    if 'm_total_rec_candidates' in df and df['m_total_rec_candidates'].notna().any():
        axes_h[0].hist(df['m_total_rec_candidates'].dropna(), bins=20, alpha=0.8)
        axes_h[0].set_title('Reco Tracks per Event (candidates)'); axes_h[0].set_xlabel('# tracks'); axes_h[0].set_ylabel('Events');
    else:
        axes_h[0].axis('off')
    # 2) ghosts per event
    if 'm_n_ghosts' in df and df['m_n_ghosts'].notna().any():
        axes_h[1].hist(df['m_n_ghosts'].dropna(), bins=20, alpha=0.8)
        axes_h[1].set_title('Ghost Tracks per Event'); axes_h[1].set_xlabel('# ghosts'); axes_h[1].set_ylabel('Events');
    else:
        axes_h[1].axis('off')
    # 3) efficiency per event
    if 'm_reconstruction_efficiency' in df and df['m_reconstruction_efficiency'].notna().any():
        axes_h[2].hist(df['m_reconstruction_efficiency'].dropna(), bins=20, range=(0,1), alpha=0.8)
        axes_h[2].set_title('Per-Event Track Efficiency'); axes_h[2].set_xlabel('Efficiency'); axes_h[2].set_ylabel('Events');
    else:
        axes_h[2].axis('off')
    plt.tight_layout()
    if save_png:
        fig_h.savefig(f"{out_prefix}_event_histograms.png", dpi=150, bbox_inches='tight')

    # ---- Scatter diagnostics ----
    fig_s, axes_s = plt.subplots(1, 2, figsize=(14, 5))
    # clones vs total truth tracks
    if 'm_total_truth_tracks' in df and 'm_n_clones' in df:
        axes_s[0].scatter(df['m_total_truth_tracks'], df['m_n_clones'], s=18, alpha=0.7)
        axes_s[0].set_xlabel('Total Truth Tracks per Event'); axes_s[0].set_ylabel('# Clones')
        axes_s[0].set_title('Clones vs Event Truth Multiplicity'); axes_s[0].grid(alpha=0.3)
    else:
        axes_s[0].axis('off')
    # purity vs total truth tracks
    pur_ev = 'm_purity_primary_only' if 'm_purity_primary_only' in df else ('m_purity_all_matched' if 'm_purity_all_matched' in df else None)
    if pur_ev is not None and 'm_total_truth_tracks' in df:
        axes_s[1].scatter(df['m_total_truth_tracks'], df[pur_ev], s=18, alpha=0.7)
        axes_s[1].set_xlabel('Total Truth Tracks per Event'); axes_s[1].set_ylabel('Track Purity (primary)')
        axes_s[1].set_title('Purity vs Event Truth Multiplicity'); axes_s[1].set_ylim(0.0, 1.05); axes_s[1].grid(alpha=0.3)
    else:
        axes_s[1].axis('off')
    plt.tight_layout()
    if save_png:
        fig_s.savefig(f"{out_prefix}_scatter_diagnostics.png", dpi=150, bbox_inches='tight')

    # ---- Heatmaps over (MS, Drop) ----
    def _heatmap(df0: pd.DataFrame, value_col: str, title: str, fname: str, cmap: str = 'viridis'):
        if df0.empty or value_col not in df0:
            return None
        piv = df0.pivot_table(index='p_multi_scatter', columns='p_drop_rate', values=value_col, aggfunc='mean')
        if piv.shape[0] < 2 or piv.shape[1] < 2:
            return None
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap=cmap,
                       extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
        ax.set_xlabel('Drop Rate'); ax.set_ylabel('Multiple Scattering'); ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label(value_col)
        plt.tight_layout()
        if save_png:
            fig.savefig(fname, dpi=150, bbox_inches='tight')
        return fig

    hm_eff = _heatmap(df, 'm_reconstruction_efficiency', 'Efficiency over (MS, Drop)', f"{out_prefix}_heatmap_eff.png")
    hm_ghost = _heatmap(df, 'm_ghost_rate', 'Ghost Rate over (MS, Drop)', f"{out_prefix}_heatmap_ghost.png")
    pur_col = 'm_purity_primary_only' if 'm_purity_primary_only' in df else ('m_purity_all_matched' if 'm_purity_all_matched' in df else None)
    hm_pur = _heatmap(df, pur_col, 'Purity over (MS, Drop)', f"{out_prefix}_heatmap_purity.png") if pur_col else None

    # ---- Sensitivities (slopes) wrt MS and Hit Resolution at baseline ----
    # Build slope tables for key rates/metrics
    sens_metrics = [
        'm_reconstruction_efficiency',
        'm_ghost_rate',
        'm_clone_fraction_total',
    ]
    if 'm_hit_efficiency_weighted' in base_ms: sens_metrics.append('m_hit_efficiency_weighted')
    if 'm_hit_efficiency_mean' in base_ms: sens_metrics.append('m_hit_efficiency_mean')
    if 'm_purity_primary_only' in base_ms: sens_metrics.append('m_purity_primary_only')
    if 'm_purity_all_matched' in base_ms: sens_metrics.append('m_purity_all_matched')

    # Slopes vs MS for each hit_res
    sens_vs_ms_rows = []
    for metric in sens_metrics:
        cur = aggregate_mean_rms(base_ms, ['p_hit_res', 'p_multi_scatter'], metric)
        for hr, sub in cur.groupby('p_hit_res'):
            slope = _finite_slope(sub['p_multi_scatter'].to_numpy(), sub['mean'].to_numpy())
            sens_vs_ms_rows.append({'metric': metric, 'p_hit_res': hr, 'slope_vs_ms': slope})
    sens_vs_ms = pd.DataFrame(sens_vs_ms_rows)

    # Slopes vs Hit Resolution for each MS
    sens_vs_hr_rows = []
    for metric in sens_metrics:
        cur = aggregate_mean_rms(base_ms, ['p_multi_scatter', 'p_hit_res'], metric)
        for ms, sub in cur.groupby('p_multi_scatter'):
            slope = _finite_slope(sub['p_hit_res'].to_numpy(), sub['mean'].to_numpy())
            sens_vs_hr_rows.append({'metric': metric, 'p_multi_scatter': ms, 'slope_vs_hit_res': slope})
    sens_vs_hitres = pd.DataFrame(sens_vs_hr_rows)

    # Plot sensitivities as tidy bar charts
    if not sens_vs_ms.empty:
        nmet = sens_vs_ms['metric'].nunique()
        fig_s1, axes_s1 = plt.subplots(nmet, 1, figsize=(9, 3.2*nmet), squeeze=False)
        for i, (metric, sub) in enumerate(sens_vs_ms.groupby('metric')):
            axes_s1[i,0].bar(sub['p_hit_res'].astype(str), sub['slope_vs_ms'])
            axes_s1[i,0].set_title(f"Sensitivity d({metric})/d(MS) at baseline")
            axes_s1[i,0].set_xlabel('Hit Resolution'); axes_s1[i,0].set_ylabel('slope')
            axes_s1[i,0].grid(alpha=0.3)
        plt.tight_layout()
        if save_png:
            fig_s1.savefig(f"{out_prefix}_sensitivities_vs_ms.png", dpi=150, bbox_inches='tight')
        if save_csv:
            sens_vs_ms.to_csv(f"{out_prefix}_sensitivities_vs_ms.csv", index=False)

    if not sens_vs_hitres.empty:
        nmet = sens_vs_hitres['metric'].nunique()
        fig_s2, axes_s2 = plt.subplots(nmet, 1, figsize=(9, 3.2*nmet), squeeze=False)
        for i, (metric, sub) in enumerate(sens_vs_hitres.groupby('metric')):
            axes_s2[i,0].bar(sub['p_multi_scatter'].astype(str), sub['slope_vs_hit_res'])
            axes_s2[i,0].set_title(f"Sensitivity d({metric})/d(hit_res) at baseline")
            axes_s2[i,0].set_xlabel('Multiple Scattering'); axes_s2[i,0].set_ylabel('slope')
            axes_s2[i,0].grid(alpha=0.3)
        plt.tight_layout()
        if save_png:
            fig_s2.savefig(f"{out_prefix}_sensitivities_vs_hit_res.png", dpi=150, bbox_inches='tight')
        if save_csv:
            sens_vs_hitres.to_csv(f"{out_prefix}_sensitivities_vs_hit_res.csv", index=False)

    # ---- Effect of epsilon window (if present) ----
    if 'p_epsilon' in df and df['p_epsilon'].notna().any():
        df_eps = df.copy()
        _to_numeric(df_eps, ['p_epsilon'])
        # Efficiency vs epsilon (baseline)
        eff_eps = aggregate_mean_rms(df_eps[np.isclose(df_eps['p_drop_rate'].to_numpy(dtype=float), 0.0, atol=drop_zero_tol)],
                                     ['p_hit_res', 'p_epsilon', 'p_multi_scatter'], 'm_reconstruction_efficiency')
        # Collapse over MS to mean per (hit_res, epsilon)
        if not eff_eps.empty:
            eff_eps2 = eff_eps.groupby(['p_hit_res','p_epsilon'], dropna=False)['mean'].agg(['mean']).reset_index().rename(columns={'mean':'mean_mean'})
            fig_eps, ax_eps = plt.subplots(1, 1, figsize=(7.5, 5))
            for hr, sub in eff_eps2.groupby('p_hit_res'):
                sub = sub.sort_values('p_epsilon')
                ax_eps.plot(sub['p_epsilon'], sub['mean_mean'], marker='o', label=f'p_hit_res={hr}')
            ax_eps.set_xlabel('epsilon window'); ax_eps.set_ylabel('Track Efficiency (mean over MS)');
            ax_eps.set_title('Effect of epsilon on Efficiency (baseline)'); ax_eps.grid(alpha=0.3); ax_eps.legend()
            if save_png:
                fig_eps.savefig(f"{out_prefix}_epsilon_efficiency.png", dpi=150, bbox_inches='tight')
        # Ghost vs epsilon
        ghost_eps = aggregate_mean_rms(df_eps[np.isclose(df_eps['p_drop_rate'].to_numpy(dtype=float), 0.0, atol=drop_zero_tol)],
                                       ['p_hit_res', 'p_epsilon', 'p_multi_scatter'], 'm_ghost_rate')
        if not ghost_eps.empty:
            ghost_eps2 = ghost_eps.groupby(['p_hit_res','p_epsilon'], dropna=False)['mean'].agg(['mean']).reset_index().rename(columns={'mean':'mean_mean'})
            fig_epsg, ax_epsg = plt.subplots(1, 1, figsize=(7.5, 5))
            for hr, sub in ghost_eps2.groupby('p_hit_res'):
                sub = sub.sort_values('p_epsilon')
                ax_epsg.plot(sub['p_epsilon'], sub['mean_mean'], marker='o', label=f'p_hit_res={hr}')
            ax_epsg.set_xlabel('epsilon window'); ax_epsg.set_ylabel('Ghost Rate (mean over MS)');
            ax_epsg.set_title('Effect of epsilon on Ghost Rate (baseline)'); ax_epsg.grid(alpha=0.3); ax_epsg.legend()
            if save_png:
                fig_epsg.savefig(f"{out_prefix}_epsilon_ghost.png", dpi=150, bbox_inches='tight')

    # ---- Optional purity-thresholded repeats (documentation note) ----
    if purity_min is not None:
        _warn_purity_cut_posthoc(df, purity_min)


    # ---- NEW: Grid of all metrics vs Multiple Scattering (baseline) ----
    metric_specs = [
        ('m_reconstruction_efficiency', 'Track Efficiency', (0.0, 1.05)),
        ('m_ghost_rate', 'Ghost Rate', None),
        ('m_clone_fraction_total', 'Clone Fraction (total)', None),
        ('m_clone_fraction_among_matched', 'Clone Fraction (among matched)', None),
        ('m_purity_primary_only', 'Track Purity (primary)', (0.0, 1.05) if 'm_purity_primary_only' in df else None),
        ('m_purity_all_matched', 'Track Purity (all matched)', (0.0, 1.05)),
        ('m_hit_efficiency_weighted', 'Hit Efficiency (primary, |T|-weighted)', (0.0, 1.05) if 'm_hit_efficiency_weighted' in df else None),
        ('m_hit_efficiency_mean', 'Hit Efficiency (primary, mean)', (0.0, 1.05)),
    ]
    present_specs = [(col, label, ylims) for (col, label, ylims) in metric_specs if col in base_ms.columns]
    nplots = len(present_specs)
    if nplots > 0:
        ncols = 3
        nrows = int(np.ceil(nplots / ncols))
        figg, axesg = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 4.8*nrows))
        axesg = np.atleast_1d(axesg).ravel()
        for i, (col, label, ylims) in enumerate(present_specs):
            agg = aggregate_mean_rms(base_ms, ['p_hit_res', 'p_multi_scatter'], col)
            # auto y-lims for rates in [0,1]
            if ylims is None and not agg.empty and agg['mean'].notna().any():
                ymax = float(max(0.05, agg['mean'].max() * 1.25))
                ylims = (0.0, min(1.05, ymax))
            _plot_errcurves(
                axesg[i], agg, x_col='p_multi_scatter', label_col='p_hit_res',
                xlabel='Multiple Scattering Parameter', ylabel=label,
                title=f'{label} vs Multiple Scattering ({base_tag})',
                x_values=ms_values, y_lims=ylims,
            )
            if save_csv:
                agg.to_csv(f"{out_prefix}_{col}_vs_ms.csv", index=False)
        for j in range(i+1, len(axesg)):
            axesg[j].axis('off')
        plt.tight_layout()
        if save_png:
            figg.savefig(f"{out_prefix}_ALL_METRICS_vs_ms_grid.png", dpi=150, bbox_inches='tight')

    return {
        'clone_vs_ms': clone_agg,
        'eff_vs_hit_res': eff_vs_res,
        'event_histos': True,
        'scatter_diags': True,
        'heatmaps': bool(hm_eff or hm_ghost or hm_pur),
        'baseline': base_tag,
        'all_metrics_vs_ms': nplots,
        'sensitivities_files': ['sensitivities_vs_ms.png', 'sensitivities_vs_hit_res.png'],
    }

# ============================= CLI =============================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='LHCb-style tracking plots with RMS error bars.')
    ap.add_argument('--csv', type=str, required=True, help='Path to input CSV with results.')
    ap.add_argument('--out', type=str, default='perf', help='Output file prefix for figures/CSVs.')
    ap.add_argument('--ms_fixed', type=float, default=2e-4, help='Multiple scattering value to hold fixed for drop-rate scans.')
    ap.add_argument('--drop_zero_tol', type=float, default=1e-9, help='Tolerance for treating drop rate as zero for baseline plots.')
    ap.add_argument('--ms_values', type=float, nargs='*', help='Explicit MS x-values (three numbers).')
    ap.add_argument('--drop_values', type=float, nargs='*', help='Explicit drop-rate x-values (three numbers).')
    ap.add_argument('--hit_res_values', type=float, nargs='*', help='Explicit hit-res curve labels (three numbers).')
    # Thresholded plots using per-track tables
    ap.add_argument('--tracks_dir', type=str, default=None,
                    help='Optional directory of per-event track tables (parquet/csv). If provided, purity-thresholded plots will be produced.')
    ap.add_argument('--purity_min', type=float, default=0.7, help='Purity threshold for thresholded plots (requires per-track tables).')
    ap.add_argument('--completeness_min', type=float, default=None, help='Optional completeness gate for thresholded plots.')
    ap.add_argument('--min_shared_hits', type=int, default=0, help='Optional |R∩T| gate for thresholded plots.')
    ap.add_argument('--no_png', action='store_true', help='Do not save PNG figures.')
    ap.add_argument('--no_csv', action='store_true', help='Do not save aggregated CSV tables.')
    ap.add_argument('--no_scatter', action='store_true', help='Do not generate the efficiency-vs-ghost scatter plot.')
    return ap.parse_args()


def main():
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    plot_all(
        df, out_prefix=args.out, ms_fixed=float(args.ms_fixed),
        drop_zero_tol=float(args.drop_zero_tol), save_png=not args.no_png,
        save_csv=not args.no_csv, extra_scatter=not args.no_scatter,
        ms_values=args.ms_values, drop_values=args.drop_values, hit_res_values=args.hit_res_values,
    )

    plot_all_extended(
        df, out_prefix=f"{args.out}_ext", ms_fixed=float(args.ms_fixed),
        drop_zero_tol=float(args.drop_zero_tol), save_png=not args.no_png,
        save_csv=not args.no_csv, ms_values=args.ms_values,
        drop_values=args.drop_values, hit_res_values=args.hit_res_values,
        purity_min=args.purity_min,
    )

    # Optional thresholded plots using per-track tables
    if args.tracks_dir is not None:
        try:
            generate_thresholded_plots(df, Path(args.tracks_dir), purity_min=args.purity_min,
                                       completeness_min=args.completeness_min,
                                       min_shared_hits=args.min_shared_hits,
                                       out_prefix=f"{args.out}_thr",
                                       ms_values=args.ms_values, drop_values=args.drop_values,
                                       hit_res_values=args.hit_res_values,
                                       save_png=not args.no_png, save_csv=not args.no_csv,
                                       drop_zero_tol=float(args.drop_zero_tol))
        except Exception as e:
            print(f"[warning] Thresholded plots skipped: {e}")


if __name__ == '__main__':
    main()
