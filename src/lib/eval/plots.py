"""Report figures. All take tidy pandas DataFrames written by the evaluation stage."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POSITION_ORDER = ["pre3", "pre2", "pre1", "post"]


def layer_curve(df, path, model="logreg", split="test"):
    """AUROC per layer, one line per probe position."""
    d = df[(df.model == model) & (df.split == split)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for pos in POSITION_ORDER:
        p = d[d.position == pos].sort_values("layer")
        if len(p):
            ax.plot(p.layer, p.auroc, marker="o", label=pos)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("layer")
    ax.set_ylabel(f"{split} AUROC")
    ax.legend(title="position")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def early_warning_curve(df, path):
    """Best-layer AUROC against how many tokens before the object word the probe reads."""
    order = {p: k for k, p in enumerate(reversed(POSITION_ORDER))}  # post=0, pre1=1, ...
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (model, excl), g in df.groupby(["model", "exclude_overlap"]):
        g = g.assign(k=g.position.map(order)).sort_values("k")
        label = f"{model}" + (" (no overlap)" if excl else "")
        ax.errorbar(g.k, g.auroc, yerr=g.get("auroc_std"), marker="o", capsize=3, label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["post (0)", "pre1", "pre2", "pre3"])
    ax.set_xlabel("tokens before the object word")
    ax.set_ylabel("test AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def bar(df, x, y, path, ylabel=None, err=None):
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(df)), 3.5))
    ax.bar(df[x], df[y], yerr=df[err] if err else None, capsize=3)
    ax.set_ylabel(ylabel or y)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
