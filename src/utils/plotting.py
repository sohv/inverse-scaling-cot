import logging
import math

LOGGER = logging.getLogger(__name__)

LEGEND_ENTRY_THRESHOLD = 5


def legend_below(fig, threshold: int = LEGEND_ENTRY_THRESHOLD, pad: float = 0.04) -> None:
    """Move a crowded legend out of the axes and under the x-axis label.

    No-op unless some axes carries at least `threshold` entries. Panel legends are merged into
    one figure-level legend, deduplicated by label, since repeated panels share their series.
    """
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    if not legends or max(len(leg.get_texts()) for leg in legends) < threshold: return

    handles, labels = [], []
    for ax in fig.axes:
        if ax.get_legend() is None: continue
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    for leg in legends:
        leg.remove()

    ncol = min(5, math.ceil(len(labels) / 2))
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)
    reserve_legend_space(fig, legend, pad)


def reserve_legend_space(fig, legend, pad: float = 0.04) -> None:
    """Shrink the axes upward so the legend sits clear of the tick labels and x-axis label."""
    fig.canvas.draw()
    height = legend.get_window_extent(fig.canvas.get_renderer()).height / (fig.get_figheight() * fig.dpi)
    # tight layout installs an engine that re-runs at draw time and would reclaim the strip
    fig.set_layout_engine("none")
    fig.subplots_adjust(bottom=min(0.5, fig.subplotpars.bottom + height + pad))
