"""The LaTeX for the figures and tables of the paper: TikZ bar charts and booktabs tables.

This module only draws. The figures.py files select the data.
The charts are lists of TikZ lines, not one .format() template. A template would need two of each TikZ brace.
"""

import math
from dataclasses import dataclass
from pathlib import Path

Cell = tuple[float, float, float]  # (low, point, high)

# The bar colours of all charts: name -> hex.
PALETTE = {
    "barexpert": "6B6B6B",
    "barone": "2A78D6",
    "bartwo": "EB6834",
    "barthree": "1BAF7A",
    "barfour": "8E5EA2",
}


@dataclass
class Cluster:
    """One group of bars, with one bar for each series.

    `bars[s]` has one cell for a bar, or no cell for a series with no bar in this cluster.
    """

    title: str  # the label under the cluster; "\\\\" starts a new line
    bars: list[list[Cell]]


def bar_positions(n_clusters: int, n_bars: int) -> list[list[float]]:
    """The x centre of each bar, with one list for each cluster.

    The bars in a cluster are 0.62 apart. The gap between two clusters is 0.54.
    """
    spacing = 0.62 * n_bars + 0.54
    return [
        [
            1.00 + spacing * cluster + 0.62 * (bar - (n_bars - 1) / 2)
            for bar in range(n_bars)
        ]
        for cluster in range(n_clusters)
    ]


def nice_scale(low: float, high: float) -> tuple[float, float, float]:
    """A rounded (low, high, step) for an axis, with eight ticks or fewer.

    The axis always includes zero and all of the data.
    """
    low, high = min(0.0, low), max(0.0, high)
    for step in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        rounded_low, rounded_high = (
            math.floor(low / step) * step,
            math.ceil(high / step) * step,
        )
        if rounded_high == rounded_low:
            rounded_high = rounded_low + step
        if (rounded_high - rounded_low) / step <= 8:
            return rounded_low, rounded_high, step
    raise AssertionError(
        f"no step in the list gives a readable axis for [{low}, {high}]"
    )


def caption_of(latex: str) -> str | None:
    r"""The text of the first \caption{...} in `latex`. None if there is no caption.

    The function counts braces, because a caption can contain other groups such as \emph{...}.
    It steps over escaped characters such as \{ and \%. It ignores text after an unescaped %.
    """
    marker = "\\caption{"
    start = latex.find(marker)
    if start == -1:
        return None
    body = start + len(marker)
    depth, commented, i = 1, False, body
    while i < len(latex):
        character = latex[i]
        if character == "\n":
            commented = False
        elif commented:
            pass
        elif character == "\\":
            i += 2
            continue
        elif character == "%":
            commented = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return latex[body:i]
        i += 1
    return None


def write(directory: Path, stem: str, text: str, draft_order: dict[str, str]) -> None:
    """Write the float to "<position>_<stem>.tex" and keep the caption that is in the file now.

    The generated text has an empty \\caption{}. If the file exists, its caption goes into the new text.
    The function prints "new", "unchanged" or "CHANGED". If a number changes, examine the caption manually.
    If `stem` has no position in `draft_order`, the function does not write the file.
    """
    if stem not in draft_order:
        print(
            f"{'unused':<10} {stem} has no position in the draft, so it is not written"
        )
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{draft_order[stem]}_{stem}.tex"

    kept = caption_of(latex=path.read_text()) if path.exists() else None
    if kept:
        text = text.replace("\\caption{}", f"\\caption{{{kept}}}", 1)

    state = (
        "new"
        if not path.exists()
        else ("unchanged" if path.read_text() == text else "CHANGED")
    )
    path.write_text(text)
    print(
        f"{state:<10} {path.name}{'' if kept is None else f', caption kept ({len(kept)} chars)'}"
    )


def legend(
    series: list[str],
    colours: list[str],
    series_label: str,
    y: float,
    raise_cm: float = 0.45,
) -> list[str]:
    """The legend row above a chart: the label, then a colour key and a name for each series.

    Each entry starts a fixed gap after the previous name. Thus, long names do not touch.
    The row starts at x = 1.5, `raise_cm` above `y`.
    """
    lines = [
        f"\t\t\\node[glabel, anchor=east] at ([yshift={raise_cm:.2f}cm]1.35,{y:.3f}) {{{series_label}}};"
    ]
    for slot, (name, colour) in enumerate(zip(series, colours)):
        at = (
            f"([yshift={raise_cm:.2f}cm]1.50,{y:.3f})"
            if slot == 0
            else f"([xshift=0.45cm]legendname{slot - 1}.east)"
        )
        lines += [
            f"\t\t\\node[key, fill={colour}!85, draw={colour}!70!black, anchor=center] (legendkey{slot}) at {at} {{}};",
            f"\t\t\\node[anchor=west, inner sep=2pt] (legendname{slot}) at (legendkey{slot}.east) {{{name}}};",
        ]
    return lines


def cluster_bar_chart(
    clusters: list[Cluster],
    series: list[str],
    colours: list[str],
    series_label: str,
    y_label: str,
    x_unit_cm: float,
    label: str,
    scale: float = 1.0,
    placement: str = "t",
    y_label_horizontal: bool = False,
    brackets: list[tuple[int, int, int]] | None = None,
    cluster_groups: list[tuple[str, int, int]] | None = None,
) -> str:
    r"""A figure with one grouped bar chart. Each bar has an error bar.

    The caption is always empty. The authors write the captions, and write() keeps them.
    `series` are the legend names. `colours` gives a palette name for each series.
    An empty cell list keeps the place of a bar but draws nothing.
    `x_unit_cm` is the width of one x unit. Make it smaller to fit more clusters.
    `scale` changes the size of the full picture, text included.
    The y axis comes from the data. The plot area is 5 cm tall.
    `placement` is the float specifier. `y_label_horizontal` puts the y label level.
    `brackets` are (cluster, bar, bar) triples. Each draws a bracket with a star over two bars with a significant difference.
    `cluster_groups` are (title, first cluster, last cluster) triples. Each draws a header under the cluster labels.
    """
    assert len(series) == len(colours) and set(colours) <= set(PALETTE), (
        f"one palette name per series, from {sorted(PALETTE)}"
    )
    assert all(len(cluster.bars) == len(series) for cluster in clusters), (
        "every cluster needs one bar per series"
    )
    assert all(len(cells) <= 1 for cluster in clusters for cells in cluster.bars), (
        "a bar has one cell or none"
    )

    cells = [cell for cluster in clusters for bars in cluster.bars for cell in bars]
    low, high, step = nice_scale(
        low=min((cell[0] for cell in cells), default=0.0),
        high=max((cell[2] for cell in cells), default=0.2),
    )

    def in_cm(value: float) -> float:
        """The height of a value in cm."""
        return 5.0 * (value - low) / (high - low)

    positions = bar_positions(n_clusters=len(clusters), n_bars=len(series))
    # Put the cluster title over the bars that the cluster has.
    drawn = [
        [x for x, cells in zip(centres, cluster.bars) if cells]
        for cluster, centres in zip(clusters, positions)
    ]
    centre_of = [sum(shown) / len(shown) for shown in drawn]

    bars = []
    for cluster, centres in zip(clusters, positions):
        for cells, x, colour in zip(cluster.bars, centres, colours):
            for cell_low, point, cell_high in cells:
                bars.append(
                    f"\t\t\t\t{x:.2f}/{in_cm(value=cell_low):.3f}/{in_cm(value=point):.3f}"
                    f"/{in_cm(value=cell_high):.3f}/{colour}/0.28/85/{in_cm(value=0.0):.3f}"
                )
    lines = [
        *[
            rf"\providecolor{{{name}}}{{HTML}}{{{hex_}}}"
            for name, hex_ in PALETTE.items()
        ],
        "",
        # [t] puts the figure at the top of a page, so it does not split a paragraph.
        rf"\begin{{figure}}[{placement}]",
        "\t\\centering",
        "\t\\begin{tikzpicture}[",
        f"\t\t\tx={x_unit_cm}cm, y=1cm,",
        *(
            [f"\t\t\tscale={scale}, every node/.append style={{scale={scale}}},"]
            if scale != 1.0
            else []
        ),
        "\t\t\tfont=\\footnotesize,",
        "\t\t\tgrid/.style={draw=black!12, line width=0.3pt},",
        "\t\t\tzeroline/.style={draw=black!55, line width=0.5pt},",
        "\t\t\taxisline/.style={draw=black!45, line width=0.4pt},",
        "\t\t\terrbar/.style={draw=black!70, line width=0.55pt},",
        "\t\t\ttick/.style={font=\\scriptsize, text=black!65},",
        "\t\t\tglabel/.style={font=\\scriptsize, text=black!55},",
        "\t\t\tkey/.style={draw=black!25, minimum width=0.26cm, minimum height=0.16cm, inner sep=0pt, line width=0.3pt},",
        "\t\t]",
        "",
    ]
    span_left, span_right = positions[0][0] - 0.28, positions[-1][-1] + 0.28
    for k in range(int(round((high - low) / step)) + 1):
        value = low + step * k
        tick_text = (
            f"{value:.2f}".rstrip("0").rstrip(".") if step < 1 else f"{value:.0f}"
        )
        lines.append(
            f"\t\t\\draw[grid] ({span_left:.2f},{in_cm(value=value):.3f}) -- ({span_right:.2f},{in_cm(value=value):.3f});"
        )
        lines.append(
            f"\t\t\\node[tick, anchor=east] at (-0.10,{in_cm(value=value):.3f}) {{{tick_text}}};"
        )
    lines.append("\t\t\\draw[axisline] (0.00,0) -- (0.00,5);")
    lines.append(
        f"\t\t\\draw[zeroline] ({span_left:.2f},{in_cm(value=0.0):.3f}) -- ({span_right:.2f},{in_cm(value=0.0):.3f});"
    )
    if y_label_horizontal:
        lines.append(
            f"\t\t\\node[anchor=east, text=black!70, xshift=-0.6cm] at (-0.10,2.5) {{{y_label}}};"
        )
    else:
        lines.append(
            f"\t\t\\node[rotate=90, anchor=south, text=black!70] at (-0.88,2.5) {{{y_label}}};"
        )
    lines += [
        "",
        # Each bar is x centre / low / point / high / colour / half width / shade / baseline, in cm.
        "\t\t\\foreach \\x/\\lo/\\m/\\hi/\\c/\\w/\\s/\\b in {%",
        ",\n".join(bars) + "}{%",
        "\t\t\t\t\\fill[\\c!\\s] ({\\x-\\w},\\b) rectangle ({\\x+\\w},\\m);",
        "\t\t\t\t\\draw[draw=\\c!70!black, line width=0.3pt] ({\\x-\\w},\\b) rectangle ({\\x+\\w},\\m);",
        "\t\t\t\t\\draw[errbar] (\\x,\\lo) -- (\\x,\\hi);",
        "\t\t\t\t\\draw[errbar] ({\\x-0.36*\\w},\\lo) -- ({\\x+0.36*\\w},\\lo);",
        "\t\t\t\t\\draw[errbar] ({\\x-0.36*\\w},\\hi) -- ({\\x+0.36*\\w},\\hi);}",
        "",
    ]
    top_bracket = None
    if brackets:
        placed = []  # (cluster, left, right, height), in the order drawn
        for index, first, second in sorted(
            brackets, key=lambda b: (b[0], b[2] - b[1], b[1])
        ):
            left, right = (
                positions[index][first] + 0.04,
                positions[index][second] - 0.04,
            )
            height = (
                max(
                    in_cm(value=cell[2])
                    for bars_ in clusters[index].bars[first : second + 1]
                    for cell in bars_
                )
                + 0.125
            )
            for other, l, r, h in placed:
                if other == index and l <= right and left <= r:
                    height = max(height, h + 0.28)
            placed.append((index, left, right, height))
        top_bracket = max(h for *_, h in placed)
        lines += [
            # Each bracket is left x / right x / height, in cm.
            "\t\t\\foreach \\l/\\r/\\y in {%",
            ",\n".join(f"\t\t\t\t{l:.2f}/{r:.2f}/{h:.2f}" for _, l, r, h in placed)
            + "}{%",
            "\t\t\t\t\\draw[draw=black!70, line width=0.4pt] (\\l,\\y-0.07) -- (\\l,\\y) -- (\\r,\\y) -- (\\r,\\y-0.07);",
            "\t\t\t\t\\node[anchor=base, inner sep=0pt, font=\\small] at ({(\\l+\\r)/2},\\y-0.02) {$*$};}",
            "",
        ]
    for x, cluster in zip(centre_of, clusters):
        lines.append(
            f"\t\t\\node[anchor=north, align=center, inner sep=3pt] at ({x:.2f},0) {{{cluster.title}}};"
        )
    if cluster_groups:
        spans = [
            (title, positions[first][0] - 0.18, positions[last][-1] + 0.18)
            for title, first, last in cluster_groups
        ]
        lines += [
            f"\t\t\\draw[black!35] ({l:.2f},-0.95) -- ({r:.2f},-0.95);"
            for _, l, r in spans
        ]
        lines += [
            f"\t\t\\node[anchor=north, font=\\footnotesize\\bfseries] at ({(l + r) / 2:.2f},-0.98) {{{title}}};"
            for title, l, r in spans
        ]

    lines += [""]
    # The legend is 0.45 cm above the plot, or 0.5 cm above the highest bracket.
    lines += legend(
        series=series,
        colours=colours,
        series_label=series_label,
        y=5,
        raise_cm=0.45 if top_bracket is None else max(0.45, top_bracket - 5 + 0.5),
    )
    lines += [
        "",
        "\t\\end{tikzpicture}",
        r"	\caption{}",
        f"\t\\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)


def booktabs_table(
    first_header: str,
    columns: list[str],
    rows: list[list[str] | str],
    group_header: str | list[tuple[str, int]] | None,
    label: str,
) -> str:
    r"""A booktabs table. It has one `l` column for the row labels and one `c` column for each entry of `columns`.

    A list row is the row label and its cells. A string row goes into the table with no change, for example a \midrule.
    A string `group_header` is one title over all data columns. A list of (title, width) pairs gives one title for each group of columns.
    The caption is always empty, as in cluster_bar_chart.
    The cells are strings. The caller formats them.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
    ]
    if isinstance(group_header, str):
        lines += [
            rf" & \multicolumn{{{len(columns)}}}{{c}}{{{group_header}}} \\",
            rf"\cmidrule(lr){{2-{len(columns) + 1}}}",
        ]
    elif group_header is not None:
        assert sum(width for _, width in group_header) == len(columns), (
            "the groups must span every data column"
        )
        spans, rules, start = [], [], 2
        for title, width in group_header:
            spans.append(rf"\multicolumn{{{width}}}{{c}}{{{title}}}")
            if title:  # a group with no title gets no rule
                rules.append(rf"\cmidrule(lr){{{start}-{start + width - 1}}}")
            start += width
        lines += [" & " + " & ".join(spans) + r" \\", " ".join(rules)]
    lines += [f"{first_header} & " + " & ".join(columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


@dataclass
class Panel:
    """One chart in a stack. It has its own y label and clusters. All charts in the stack use the same x axis."""

    y_label: str
    clusters: list[Cluster]


def stacked_cluster_bar_charts(
    panels: list[Panel],
    series: list[str],
    colours: list[str],
    series_label: str,
    x_unit_cm: float,
    panel_height_cm: float,
    gap_cm: float,
    label: str,
) -> str:
    r"""Many bar charts in one figure, one above the other. They use one legend, one x axis and one caption.

    One float is easier for LaTeX to place than many floats.
    Each panel has its own y axis.
    `panel_height_cm` is the height of the plot area of one panel.
    Each y label is level, to the left of the ticks. "\\" starts a new line.
    The caption is always empty, as in cluster_bar_chart.
    """
    assert len(series) == len(colours) and set(colours) <= set(PALETTE), (
        f"one palette name per series, from {sorted(PALETTE)}"
    )
    assert len({tuple(c.title for c in panel.clusters) for panel in panels}) == 1, (
        "a shared x axis needs every panel over the same clusters, in the same order"
    )
    assert all(
        len(cluster.bars) == len(series)
        for panel in panels
        for cluster in panel.clusters
    ), "every cluster needs one bar per series"

    scales = []
    for panel in panels:
        cells = [
            cell for cluster in panel.clusters for bars in cluster.bars for cell in bars
        ]
        scales.append(
            nice_scale(
                low=min((cell[0] for cell in cells), default=0.0),
                high=max((cell[2] for cell in cells), default=0.2),
            )
        )

    # Panel 0 is at the top. All values are in cm, so one \foreach draws all bars.
    def base_of(index: int) -> float:
        return (len(panels) - 1 - index) * (panel_height_cm + gap_cm)

    def in_cm(value: float, index: int) -> float:
        low, high, _ = scales[index]
        return base_of(index=index) + panel_height_cm * (value - low) / (high - low)

    positions = bar_positions(n_clusters=len(panels[0].clusters), n_bars=len(series))
    top = base_of(index=0) + panel_height_cm
    span_left, span_right = positions[0][0] - 0.28, positions[-1][-1] + 0.28

    lines = [
        *[
            rf"\providecolor{{{name}}}{{HTML}}{{{hex_}}}"
            for name, hex_ in PALETTE.items()
        ],
        "",
        # [!t], because the stack is taller than \topfraction. Without "!", LaTeX puts it on a page of its own.
        r"\begin{figure}[!t]",
        "\t\\centering",
        "\t\\begin{tikzpicture}[",
        f"\t\t\tx={x_unit_cm}cm, y=1cm,",
        "\t\t\tfont=\\footnotesize,",
        "\t\t\tgrid/.style={draw=black!12, line width=0.3pt},",
        "\t\t\tzeroline/.style={draw=black!55, line width=0.5pt},",
        "\t\t\taxisline/.style={draw=black!45, line width=0.4pt},",
        "\t\t\terrbar/.style={draw=black!70, line width=0.55pt},",
        "\t\t\ttick/.style={font=\\scriptsize, text=black!65},",
        "\t\t\tglabel/.style={font=\\scriptsize, text=black!55},",
        "\t\t\tkey/.style={draw=black!25, minimum width=0.26cm, minimum height=0.16cm, inner sep=0pt, line width=0.3pt},",
        "\t\t]",
    ]

    bars = []
    for index, panel in enumerate(panels):
        low, high, step = scales[index]
        base = base_of(index=index)
        lines += [""]
        for k in range(int(round((high - low) / step)) + 1):
            value = low + step * k
            at_cm = base + panel_height_cm * (value - low) / (high - low)
            tick_text = (
                f"{value:.2f}".rstrip("0").rstrip(".") if step < 1 else f"{value:.0f}"
            )
            lines.append(
                f"\t\t\\draw[grid] ({span_left:.2f},{at_cm:.3f}) -- ({span_right:.2f},{at_cm:.3f});"
            )
            lines.append(
                f"\t\t\\node[tick, anchor=east] at (-0.10,{at_cm:.3f}) {{{tick_text}}};"
            )
        lines.append(
            f"\t\t\\draw[axisline] (0,{base:.3f}) -- (0,{base + panel_height_cm:.3f});"
        )
        lines.append(
            f"\t\t\\draw[zeroline] ({span_left:.2f},{in_cm(value=0.0, index=index):.3f}) -- ({span_right:.2f},{in_cm(value=0.0, index=index):.3f});"
        )
        # 0.6 cm is more than the widest tick label, "-0.4".
        lines.append(
            f"\t\t\\node[anchor=east, text=black!70, xshift=-0.6cm, align=center] at (-0.10,{base + panel_height_cm / 2:.3f}) {{{panel.y_label}}};"
        )
        for cluster, centres in zip(panel.clusters, positions):
            zero = in_cm(value=0.0, index=index)
            for cells, x, colour in zip(cluster.bars, centres, colours):
                for low_, point, high_ in cells:
                    bars.append(
                        f"\t\t\t\t{x:.2f}/{in_cm(value=low_, index=index):.3f}/{in_cm(value=point, index=index):.3f}"
                        f"/{in_cm(value=high_, index=index):.3f}/{colour}/0.28/85/{zero:.3f}"
                    )

    lines += [
        "",
        # Each bar is x centre / low / point / high / colour / half width / shade / baseline, in cm.
        "\t\t\\foreach \\x/\\lo/\\m/\\hi/\\c/\\w/\\s/\\b in {%",
        ",\n".join(bars) + "}{%",
        "\t\t\t\t\\fill[\\c!\\s] ({\\x-\\w},\\b) rectangle ({\\x+\\w},\\m);",
        "\t\t\t\t\\draw[draw=\\c!70!black, line width=0.3pt] ({\\x-\\w},\\b) rectangle ({\\x+\\w},\\m);",
        "\t\t\t\t\\draw[errbar] (\\x,\\lo) -- (\\x,\\hi);",
        "\t\t\t\t\\draw[errbar] ({\\x-0.36*\\w},\\lo) -- ({\\x+0.36*\\w},\\lo);",
        "\t\t\t\t\\draw[errbar] ({\\x-0.36*\\w},\\hi) -- ({\\x+0.36*\\w},\\hi);}",
        "",
    ]
    # The cluster labels and the legend appear one time for the full stack.
    for centres, cluster in zip(positions, panels[0].clusters):
        lines.append(
            f"\t\t\\node[anchor=north, align=center, inner sep=3pt] at ({sum(centres) / len(centres):.2f},0) {{{cluster.title}}};"
        )

    lines += [""]
    lines += legend(series=series, colours=colours, series_label=series_label, y=top)
    lines += [
        "",
        "\t\\end{tikzpicture}",
        "\t\\caption{}",
        f"\t\\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)


@dataclass
class ShareRow:
    """One horizontal bar in each panel, and its label.

    A row with no label is a gap. A row with a label and no shares is a grey heading.
    """

    label: str
    shares: list[
        list[float]
    ]  # panel -> segment -> share; the shares of a panel add to one


@dataclass
class ShareBlock:
    """A group of rows with the same segments. The block has its own ticks and legend.

    `colours` are xcolor expressions, such as barone or black!18.
    `text_colours` are the colours of the percentages on the segments.
    """

    rows: list[ShareRow]
    segments: list[str]
    colours: list[str]
    text_colours: list[str]
    legend_label: str = ""  # the text before the legend keys


def share_bars(
    blocks: list[ShareBlock],
    panels: list[str],
    label: str,
    panel_width_cm: float = 4.3,
    label_width_cm: float = 3.3,
    row_height_cm: float = 0.46,
) -> str:
    r"""A figure of 100\% stacked horizontal bars. The panels are side by side and use the same row labels.

    The blocks go from top to bottom.
    A segment that is wide enough shows its percentage.
    The caption is always empty, as in cluster_bar_chart.
    """
    for block in blocks:
        assert len(block.segments) == len(block.colours) == len(block.text_colours), (
            "one colour and one text colour per segment"
        )
        assert all(
            len(row.shares) == len(panels)
            and all(len(p) == len(block.segments) for p in row.shares)
            for row in block.rows
            if row.label and row.shares
        ), "every row needs one share per segment in every panel"
    gap_cm = (
        0.8  # the 100% tick of a panel must not touch the 0% tick of the next panel
    )
    bar_cm = 0.62 * row_height_cm
    lefts = [label_width_cm + p * (panel_width_cm + gap_cm) for p in range(len(panels))]

    lines = [
        *[
            rf"\providecolor{{{name}}}{{HTML}}{{{hex_}}}"
            for name, hex_ in PALETTE.items()
        ],
        "",
        r"\begin{figure}[t]",
        "\t\\centering",
        "\t\\begin{tikzpicture}[x=1cm, y=1cm, font=\\footnotesize,",
        "\t\t\tpct/.style={font=\\scriptsize, inner sep=0pt},",
        "\t\t\ttick/.style={font=\\scriptsize, text=black!65},",
        "\t\t\tglabel/.style={font=\\scriptsize, text=black!55},",
        "\t\t\tkey/.style={draw=black!25, minimum width=0.26cm, minimum height=0.16cm, inner sep=0pt, line width=0.3pt},",
        "\t\t]",
        "",
    ]
    for left, title in zip(lefts, panels):
        lines.append(
            f"\t\t\\node[anchor=south, text=black!70] at ({left + panel_width_cm / 2:.2f},0.05) {{{title}}};"
        )

    top = 0.0  # the top edge of the current block; the blocks go down from 0
    for number, block in enumerate(blocks):
        bottom = top - len(block.rows) * row_height_cm
        lines += [""]
        for left in lefts:
            for k in range(5):
                x = left + panel_width_cm * k / 4
                lines.append(
                    f"\t\t\\draw[draw=black!12, line width=0.3pt] ({x:.2f},{bottom:.3f}) -- ({x:.2f},{top:.3f});"
                )
                lines.append(
                    f"\t\t\\node[tick, anchor=north] at ({x:.2f},{bottom - 0.04:.3f}) {{{25 * k}\\%}};"
                )
        for r, row in enumerate(block.rows):
            if not row.label:
                continue
            y = top - (r + 0.5) * row_height_cm
            if not row.shares:
                lines.append(
                    f"\t\t\\node[anchor=east, text=black!55] at ({label_width_cm - 0.08:.2f},{y:.3f}) {{{row.label}}};"
                )
                continue
            lines.append(
                f"\t\t\\node[anchor=east, text=black!80] at ({label_width_cm - 0.08:.2f},{y:.3f}) {{{row.label}}};"
            )
            for left, shares in zip(lefts, row.shares):
                x = left
                for share, colour, text_colour in zip(
                    shares, block.colours, block.text_colours
                ):
                    width = panel_width_cm * share
                    lines.append(
                        f"\t\t\\fill[{colour}] ({x:.3f},{y - bar_cm / 2:.3f}) rectangle ({x + width:.3f},{y + bar_cm / 2:.3f});"
                    )
                    if width >= 0.45:
                        lines.append(
                            f"\t\t\\node[pct, text={text_colour}] at ({x + width / 2:.3f},{y:.3f}) {{{100 * share:.0f}}};"
                        )
                    x += width
                lines.append(
                    f"\t\t\\draw[draw=black!35, line width=0.3pt] ({left:.3f},{y - bar_cm / 2:.3f}) rectangle ({left + panel_width_cm:.3f},{y + bar_cm / 2:.3f});"
                )
        # The legend is under the ticks of this block.
        legend_y = bottom - 0.62
        x = label_width_cm
        if block.legend_label:
            lines.append(
                f"\t\t\\node[glabel, anchor=east] at ({x - 0.08:.2f},{legend_y:.3f}) {{{block.legend_label}}};"
            )
        for name, colour in zip(block.segments, block.colours):
            lines.append(
                f"\t\t\\node[key, fill={colour}, anchor=west] at ({x:.2f},{legend_y:.3f}) {{}};"
            )
            lines.append(
                f"\t\t\\node[anchor=west, inner sep=2pt] at ({x + 0.3:.2f},{legend_y:.3f}) {{{name}}};"
            )
            x += 0.3 + 0.16 * len(name) + 0.5
        top = legend_y - 0.55
    lines += [
        "",
        "\t\\end{tikzpicture}",
        r"	\caption{}",
        f"\t\\label{{{label}}}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)
