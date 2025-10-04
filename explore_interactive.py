"""
Interactive exploration generator using Plotly.
Generates:
 - interactive scatter matrix (Plotly scatter matrix)
 - violin/kde comparisons for HZ vs non-HZ (Plotly)
 - permutation importance (bar chart)
 - partial dependence plots for top features
 - saves sampled CSV used for the plots

Usage:
  python explore_interactive.py --data cumulative_2025.10.03_09.12.20.csv --outdir exploration/interactive --sample 0.2

"""
import os
import argparse
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FEATURES = [
    "koi_prad",
    "koi_period",
    "koi_depth",
    "koi_insol",
    "koi_steff",
    "koi_srad",
]


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_and_sample(path, sample_frac=0.2, seed=42):
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    return df


def plot_scatter_matrix(df, out_dir, chunk_size=3, cell_size=220):
    """Create multiple smaller scatter-matrix HTML files by chunking FEATURES.

    Each chunk will produce a smaller NxN scatter matrix saved as
    scatter_matrix_group_{i}.html inside out_dir.
    """
    cols = [c for c in FEATURES if c in df.columns]
    if not cols:
        logging.warning("No features available for scatter matrix")
        return
    from plotly.subplots import make_subplots

    groups = df['koi_disposition'].unique() if 'koi_disposition' in df.columns else ['ALL']

    # compute reasonable axis ranges (2nd to 98th percentiles) to avoid extreme outliers
    ranges = {}
    for c in cols:
        colvals = pd.to_numeric(df[c], errors='coerce').dropna()
        if colvals.empty:
            ranges[c] = None
        else:
            lo = float(np.nanpercentile(colvals, 2))
            hi = float(np.nanpercentile(colvals, 98))
            if lo == hi:
                ranges[c] = None
            else:
                ranges[c] = (lo, hi)

    # helper to build a figure for a list of columns
    def _build_for_cols(subcols, group_idx):
        n = len(subcols)
        fig = make_subplots(rows=n, cols=n, shared_xaxes=False, shared_yaxes=False,
                            horizontal_spacing=0.04, vertical_spacing=0.04)
        for i, xi in enumerate(subcols):
            for j, yj in enumerate(subcols):
                row = i + 1
                col = j + 1
                if i == j:
                    for g in groups:
                        sub = df[df['koi_disposition'] == g] if 'koi_disposition' in df.columns else df
                        x = pd.to_numeric(sub[xi], errors='coerce').dropna()
                        if x.empty:
                            continue
                        hist = go.Histogram(x=x, nbinsx=30, name=str(g), opacity=0.6, marker=dict(line=dict(width=0)))
                        fig.add_trace(hist, row=row, col=col)
                    if ranges.get(xi) is not None:
                        fig.update_xaxes(title_text=xi, range=list(ranges[xi]), row=row, col=col)
                    else:
                        fig.update_xaxes(title_text=xi, row=row, col=col)
                else:
                    for g in groups:
                        sub = df[df['koi_disposition'] == g] if 'koi_disposition' in df.columns else df
                        common = sub[[yj, xi]].dropna()
                        if common.shape[0] == 0:
                            continue
                        fig.add_trace(go.Scattergl(x=common[yj], y=common[xi], mode='markers', marker=dict(size=4, opacity=0.6), name=str(g), showlegend=(i==0 and j==1)), row=row, col=col)
                    all_xy = df[[yj, xi]].dropna()
                    try:
                        fig.add_trace(go.Histogram2dContour(x=all_xy[yj], y=all_xy[xi], contours=dict(coloring='lines'), showscale=False, opacity=0.5), row=row, col=col)
                    except Exception:
                        pass
                    if ranges.get(yj) is not None:
                        fig.update_xaxes(title_text=yj, range=list(ranges[yj]), row=row, col=col)
                    else:
                        fig.update_xaxes(title_text=yj, row=row, col=col)
                    if ranges.get(xi) is not None:
                        fig.update_yaxes(title_text=xi, range=list(ranges[xi]), row=row, col=col)
                    else:
                        fig.update_yaxes(title_text=xi, row=row, col=col)

        fig.update_layout(height=cell_size * n, width=cell_size * n, title_text=f'Enhanced scatter matrix (group {group_idx})')
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        return fig

    # chunk cols into groups of chunk_size
    safe_mkdir(out_dir)
    chunks = [cols[i:i + chunk_size] for i in range(0, len(cols), chunk_size)]
    for idx, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        fig = _build_for_cols(chunk, idx)
        out = os.path.join(out_dir, f'scatter_matrix_group_{idx}.html')
        fig.write_html(out, include_plotlyjs='cdn')
        logging.info(f"Saved {out}")


# Note: HZ violin plots removed to keep the interactive explorer focused on the improved scatter matrix


def compute_permutation_importance(df, out):
    cols = [c for c in FEATURES if c in df.columns]
    if not cols or 'koi_disposition' not in df.columns:
        logging.warning("Skipping permutation importance: missing features or target")
        return
    # Simple train/test split and model fit for importance
    X = df[cols].fillna(0).values
    y = (df['koi_disposition'] == 'CONFIRMED').astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(Xtr, ytr)
    # Increase repeats for more stable importances
    r = permutation_importance(model, Xte, yte, n_repeats=120, random_state=42, n_jobs=1)
    importances = pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)
    fig = px.bar(x=importances.values, y=importances.index, orientation='h', title='Permutation importance')
    fig.update_layout(xaxis_title='Importance', yaxis_title='Feature')
    fig.write_html(out, include_plotlyjs='cdn')
    logging.info(f"Saved {out}")


def plot_partial_dependence(df, out, features=3):
    cols = [c for c in FEATURES if c in df.columns]
    if not cols or 'koi_disposition' not in df.columns:
        logging.warning("Skipping partial dependence: missing features or target")
        return
    X = df[cols].fillna(0).values
    y = (df['koi_disposition'] == 'CONFIRMED').astype(int).values
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    # We'll compute PDP manually: vary each feature across a grid and average model predictions
    # Denser grid for smoother PDP curves
    n_grid = 250
    features_idx = list(range(min(features, len(cols))))
    for fi in features_idx:
        f_name = cols[fi]
        vals = np.linspace(np.nanpercentile(X[:, fi], 2), np.nanpercentile(X[:, fi], 98), n_grid)
        avg_preds = []
        # baseline is the column-wise mean
        X_base = np.nanmean(X, axis=0, keepdims=True)
        for v in vals:
            X_grid = np.repeat(X_base, 32, axis=0)  # small batch to average over
            X_grid[:, fi] = v
            try:
                preds = model.predict_proba(X_grid)[:, 1]
            except Exception:
                preds = model.predict(X_grid).astype(float)
            avg_preds.append(float(np.mean(preds)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vals, y=avg_preds, mode='lines+markers', name=f_name))
        fig.update_layout(title=f'Partial dependence (approx): {f_name}', xaxis_title=f_name, yaxis_title='Predicted probability')
        fname = os.path.join(out, f'partial_dependence_{f_name}.html')
        fig.write_html(fname, include_plotlyjs='cdn')
        logging.info(f"Saved {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cumulative_2025.10.03_09.12.20.csv')
    ap.add_argument('--outdir', default='exploration/interactive')
    ap.add_argument('--sample', type=float, default=0.7)
    ap.add_argument('--cell-size', type=int, default=220, help='Size in pixels of each scatter matrix cell')
    args = ap.parse_args()

    df = load_and_sample(args.data, sample_frac=args.sample)
    safe_mkdir(args.outdir)
    # save sampled csv
    sample_csv = os.path.join(os.path.dirname(args.outdir), 'sampled.csv')
    df.to_csv(sample_csv, index=False)
    logging.info(f"Saved sampled data to {sample_csv}")

    # Scatter matrix -> produce smaller grouped matrices in the outdir
    # Allow scaling size of each subplot cell for clearer inspection
    plot_scatter_matrix(df, args.outdir, chunk_size=3, cell_size=args.cell_size)

    # HZ violin/kde (removed)

    # Permutation importance
    compute_permutation_importance(df, os.path.join(args.outdir, 'permutation_importance.html'))

    # Partial dependence (separate files)
    plot_partial_dependence(df, args.outdir, features=3)

    logging.info('Interactive exploration artifacts generated.')

if __name__ == '__main__':
    main()
