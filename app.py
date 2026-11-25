"""Process Monitoring Dashboard (Statistical Process Control).

Adapted from Plotly's open-source `dash-manufacture-spc-dashboard` sample app (MIT licence,
see LICENSE). Extended by Pawel Sogra:

* control limits estimated from a cleaned baseline using the moving-range method
  (the original used mean +/- 3 standard deviations of all data, and labelled mean +/- 1 sd
  as "specification limits");
* three run rules (beyond 3 sigma, 2 of 3 beyond 2 sigma, 8 in a row on one side of the mean);
* process capability (Cp, Cpk, Ppk) against user-editable specification limits;
* a live exception log, working Specification Settings tab, and updates for current
  Dash / pandas.
"""
import os
import pathlib

import dash
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dash_table, dcc, html

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Process Monitoring (SPC) | Pawel Sogra"
server = app.server
app.config["suppress_callback_exceptions"] = True

APP_PATH = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(APP_PATH / "data" / "spc_data.csv")

params = list(df)
metrics = params[1:]
max_length = len(df)
batch_x = df["Batch"].tolist()

BASELINE_N = 100  # Phase I: batches used to establish control limits
START_N = BASELINE_N  # live simulation starts after the baseline period
SPARK_WINDOW = 100  # batches shown in each sparkline
D2 = 1.128  # bias constant for moving ranges of size 2
ALARM_PCT = 7  # OOC% threshold for the red state (as in the original app)
CPK_TARGET = 1.33
TICK_MS = int(os.environ.get("SPC_INTERVAL_MS", 2000))
PORTFOLIO_URL = os.environ.get("PORTFOLIO_URL", "")

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"

RULE_LABELS = {
    "r1": "Beyond control limits (3 sigma)",
    "r2": "2 of 3 beyond 2 sigma",
    "r4": "8 in a row on one side of mean",
}
RULE_STYLE = {
    "r1": dict(color="rgba(244, 80, 96, 0.9)", symbol="square"),
    "r2": dict(color="rgba(255, 159, 67, 0.9)", symbol="diamond"),
    "r4": dict(color="rgba(168, 130, 255, 0.9)", symbol="triangle-up"),
}


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def baseline_limits(x):
    """Control limits from a cleaned baseline (individuals / moving-range chart).

    Sigma is estimated from the average moving range (MRbar / 1.128). Baseline points beyond
    3 sigma are treated as special causes and removed once before the limits are finalised.
    Specification limits default to mean +/- 4 sigma purely as an illustration; replace
    them with real specifications on the Specification Settings tab.
    """
    base = x.iloc[:BASELINE_N].to_numpy(dtype=float)
    for _ in range(2):
        mean = base.mean()
        sigma = np.abs(np.diff(base)).mean() / D2
        keep = np.abs(base - mean) <= 3 * sigma
        if keep.all():
            break
        base = base[keep]
    mean = float(base.mean())
    sigma = float(np.abs(np.diff(base)).mean() / D2)
    return {
        "mean": mean,
        "sigma": sigma,
        "ucl": mean + 3 * sigma,
        "lcl": mean - 3 * sigma,
        "usl": mean + 4 * sigma,
        "lsl": mean - 4 * sigma,
    }


BASE_LIMITS = {m: baseline_limits(df[m]) for m in metrics}


def _round_limits(d):
    return {k: float(f"{v:.6g}") for k, v in d.items()}


def initial_store():
    return {m: _round_limits(BASE_LIMITS[m]) for m in metrics}


def rule_flags(values, lim):
    """Boolean arrays marking points that violate each run rule."""
    x = pd.Series(values, dtype=float)
    mean, sigma = lim["mean"], lim["sigma"]
    z = (x - mean) / sigma

    r1 = (x > lim["ucl"]) | (x < lim["lcl"])

    above = (z > 2).astype(int)
    below = (z < -2).astype(int)
    r2 = ((above.rolling(3, min_periods=1).sum() >= 2) & (above == 1)) | (
        (below.rolling(3, min_periods=1).sum() >= 2) & (below == 1)
    )

    side = np.sign(x - mean)
    run_id = (side != side.shift()).cumsum()
    run_len = side.groupby(run_id).cumcount() + 1
    r4 = (run_len >= 8) & (side != 0)

    return {"r1": r1.to_numpy(), "r2": r2.to_numpy(), "r4": r4.to_numpy()}


def analyse(metric, store):
    """Rule flags and cumulative out-of-control fraction for one metric."""
    lim = store[metric]
    flags = rule_flags(df[metric], lim)
    any_flag = flags["r1"] | flags["r2"] | flags["r4"]
    ooc_cum = np.cumsum(any_flag) / np.arange(1, len(any_flag) + 1)
    return flags, any_flag, ooc_cum


def capability(metric, store, k):
    """Cp, Cpk, Ppk and % out of spec using data up to batch k."""
    lim = store[metric]
    x = df[metric].iloc[:k].to_numpy(dtype=float)
    mean = x.mean()
    sd_overall = x.std(ddof=1) if len(x) > 1 else np.nan
    sw = lim["sigma"]
    usl, lsl = lim["usl"], lim["lsl"]
    cp = (usl - lsl) / (6 * sw)
    cpk = min(usl - mean, mean - lsl) / (3 * sw)
    ppk = min(usl - mean, mean - lsl) / (3 * sd_overall) if sd_overall else np.nan
    out_spec = 100 * float(((x > usl) | (x < lsl)).mean())
    return {"cp": cp, "cpk": cpk, "ppk": ppk, "out_spec": out_spec}


def clamp_k(interval):
    return int(min(max(interval or 1, 1), max_length))


# --------------------------------------------------------------------------------------
# Layout builders
# --------------------------------------------------------------------------------------
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Process Monitoring Dashboard"),
                    html.H6("Control charts, run rules and capability | Pawel Sogra"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.A(
                        html.Button(children="BACK TO PORTFOLIO"),
                        href=PORTFOLIO_URL or "#",
                        style={} if PORTFOLIO_URL else {"display": "none"},
                    ),
                    html.Button(
                        id="learn-more-button", children="HOW IT WORKS", n_clicks=0
                    ),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Specification Settings",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Control Charts Dashboard",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def build_tab_1():
    return [
        html.Div(
            id="set-specs-intro-container",
            children=html.P(
                "Control limits come from the first 100 batches (a cleaned baseline). "
                "Specification limits are illustrative defaults: enter your real USL/LSL to "
                "see capability update across the dashboard."
            ),
        ),
        html.Div(
            id="settings-menu",
            children=[
                html.Div(
                    id="metric-select-menu",
                    children=[
                        html.Label(id="metric-select-title", children="Select Metrics"),
                        html.Br(),
                        dcc.Dropdown(
                            id="metric-select-dropdown",
                            options=[{"label": m, "value": m} for m in metrics],
                            value=metrics[0],
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    id="value-setter-menu",
                    children=[
                        html.Div(id="value-setter-panel", children=build_value_setter_rows()),
                        html.Br(),
                        html.Div(id="value-setter-msg", className="setter-msg"),
                        html.Div(
                            id="button-div",
                            children=[
                                html.Button(
                                    "Update", id="value-setter-set-btn", n_clicks=0
                                ),
                                html.Button(
                                    "Reset to baseline",
                                    id="value-setter-reset-btn",
                                    n_clicks=0,
                                ),
                                html.Button(
                                    "View current setup",
                                    id="value-setter-view-btn",
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        html.Div(
                            id="value-setter-view-output", className="output-datatable"
                        ),
                    ],
                ),
            ],
        ),
    ]


def build_value_setter_line(line_num, label, value, col3):
    return html.Div(
        id=line_num,
        children=[
            html.Label(label, className="four columns"),
            html.Label(value, className="four columns"),
            html.Div(col3, className="four columns"),
        ],
        className="row",
    )


def generate_modal():
    return html.Div(
        id="markdown",
        className="modal",
        children=html.Div(
            id="markdown-container",
            className="markdown-container",
            children=[
                html.Div(
                    className="close-container",
                    children=html.Button(
                        "Close",
                        id="markdown_close",
                        n_clicks=0,
                        className="closeButton",
                    ),
                ),
                html.Div(
                    className="markdown-text",
                    children=dcc.Markdown(
                        """
###### What is this?

A simulated live feed of seven process measurements, monitored with Statistical Process
Control (SPC). A new batch arrives every two seconds. Click a parameter to see its control
chart; use Stop/Start to pause the feed.

###### How limits and alarms work

- **Control limits** describe what the process actually does. They come from the first 100
  batches (baseline), with sigma estimated from the moving range, after removing obvious
  special causes.
- **Specification limits** describe what the customer requires. Here they are illustrative
  defaults; set real ones on the Specification Settings tab.
- **Alarms** use three run rules: a point beyond the control limits, 2 of 3 points beyond
  2 sigma, or 8 points in a row on one side of the mean.
- **Capability:** Cpk uses short-term (within-batch) variation, Ppk uses overall variation.
  Cpk of 1.33 or more is a common target. A big gap between Cpk and Ppk signals instability.

###### Credit

Adapted from Plotly's open-source
[dash-manufacture-spc-dashboard](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard)
(MIT licence) and extended by Pawel Sogra. The data is a demonstration dataset, not real
production data.
                        """
                    ),
                ),
            ],
        ),
    )


def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="card-1",
                children=[
                    html.P("Alerts in last 20 batches"),
                    daq.LEDDisplay(
                        id="alerts-led",
                        value="0",
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                    ),
                ],
            ),
            html.Div(
                id="card-2",
                children=[
                    html.P("Batches processed"),
                    daq.Gauge(
                        id="progress-gauge",
                        max=max_length,
                        min=0,
                        showCurrentValue=True,
                    ),
                ],
            ),
            html.Div(
                id="utility-card",
                children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
            ),
        ],
    )


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def build_top_panel(stopped_interval):
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("Process Control Metrics Summary"),
                    html.Div(
                        id="metric-div",
                        children=[
                            generate_metric_list_header(),
                            html.Div(
                                id="metric-rows",
                                children=[
                                    generate_metric_row_helper(stopped_interval, i)
                                    for i in range(1, len(params))
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="ooc-piechart-outer",
                className="four columns",
                children=[
                    generate_section_banner("Process capability (Cpk)"),
                    dcc.Graph(
                        id="capability-chart",
                        config={"displayModeBar": False},
                        style={"height": "32rem"},
                    ),
                ],
            ),
        ],
    )


def generate_metric_list_header():
    return generate_metric_row(
        "metric_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("Parameter")},
        {"id": "m_header_2", "children": html.Div("Count")},
        {"id": "m_header_3", "children": html.Div("Last 100 batches")},
        {"id": "m_header_4", "children": html.Div("OOC%")},
        {"id": "m_header_5", "children": html.Div("%OOC")},
        {"id": "m_header_6", "children": "Pass/Fail"},
    )


def sparkline_figure(metric, k):
    start = max(0, k - SPARK_WINDOW)
    return go.Figure(
        {
            "data": [
                {
                    "x": batch_x[start:k],
                    "y": df[metric].iloc[start:k].tolist(),
                    "mode": "lines+markers",
                    "name": metric,
                    "line": {"color": "#f4d44d"},
                }
            ],
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    showline=False, showgrid=False, zeroline=False, showticklabels=False
                ),
                "yaxis": dict(
                    showline=False, showgrid=False, zeroline=False, showticklabels=False
                ),
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }
    )


def generate_metric_row_helper(stopped_interval, index):
    item = params[index]
    return generate_metric_row(
        item + suffix_row,
        None,
        {
            "id": item,
            "className": "metric-row-button-text",
            "children": html.Button(
                id=item + suffix_button_id,
                className="metric-row-button",
                children=item,
                title="Click to visualize live SPC chart",
                n_clicks=0,
            ),
        },
        {"id": item + suffix_count, "children": "0"},
        {
            "id": item + "_sparkline",
            "children": dcc.Graph(
                id=item + suffix_sparkline_graph,
                style={"width": "100%", "height": "95%"},
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure=sparkline_figure(item, clamp_k(stopped_interval)),
            ),
        },
        {"id": item + suffix_ooc_n, "children": "0.00%"},
        {
            "id": item + suffix_ooc_g + "_container",
            "children": daq.GraduatedBar(
                id=item + suffix_ooc_g,
                color={
                    "ranges": {
                        "#92e0d3": [0, 3],
                        "#f4d44d ": [3, 7],
                        "#f45060": [7, 15],
                    }
                },
                showCurrentValue=False,
                max=15,
                value=0,
            ),
        },
        {
            "id": item + "_pf",
            "children": daq.Indicator(
                id=item + suffix_indicator, value=True, color="#91dfd2", size=12
            ),
        },
    )


def generate_metric_row(id, style, col1, col2, col3, col4, col5, col6):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"margin-right": "2.5rem", "minWidth": "50px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center"},
                className="one column",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%"},
                className="four columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"], style={}, className="one column", children=col4["children"]
            ),
            html.Div(
                id=col5["id"],
                style={"height": "100%", "margin-top": "5rem"},
                className="three columns",
                children=col5["children"],
            ),
            html.Div(
                id=col6["id"],
                style={"display": "flex", "justifyContent": "center"},
                className="one column",
                children=col6["children"],
            ),
        ],
    )


def build_chart_panel():
    return html.Div(
        id="control-chart-container",
        className="twelve columns",
        children=[
            generate_section_banner("Live SPC Chart"),
            dcc.Graph(
                id="control-chart-live",
                figure=go.Figure(
                    {
                        "data": [
                            {"x": [], "y": [], "mode": "lines+markers", "name": metrics[0]}
                        ],
                        "layout": {
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                            "xaxis": dict(showline=False, showgrid=False, zeroline=False),
                            "yaxis": dict(showgrid=False, showline=False, zeroline=False),
                            "autosize": True,
                        },
                    }
                ),
            ),
            generate_section_banner("Exception log (latest alarms)"),
            html.Div(id="alert-log"),
        ],
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def generate_graph(interval, store, col):
    k = clamp_k(interval)
    lim = store[col]
    flags, _, _ = analyse(col, store)
    x_array = batch_x[:k]
    y_array = df[col].iloc[:k].tolist()

    traces = [
        {
            "x": x_array,
            "y": y_array,
            "mode": "lines+markers",
            "name": col,
            "line": {"color": "#f4d44d"},
        }
    ]
    for rule in ("r1", "r2", "r4"):
        idx = np.flatnonzero(flags[rule][:k])
        # a point is shown once, under its most severe rule
        if rule != "r1":
            higher = np.zeros(k, dtype=bool)
            for hr in ("r1", "r2"):
                if hr == rule:
                    break
                higher |= flags[hr][:k]
            idx = np.flatnonzero(flags[rule][:k] & ~higher)
        traces.append(
            {
                "x": [x_array[i] for i in idx],
                "y": [y_array[i] for i in idx],
                "name": RULE_LABELS[rule],
                "mode": "markers",
                "marker": dict(size=11, **RULE_STYLE[rule]),
            }
        )
    traces.append(
        {
            "x": x_array,
            "y": y_array,
            "type": "histogram",
            "orientation": "h",
            "name": "Distribution",
            "xaxis": "x2",
            "yaxis": "y2",
            "marker": {"color": "#f4d44d"},
            "showlegend": False,
        }
    )

    def hline(y, color, width=1, dash="dot"):
        return {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": 1,
            "y0": y,
            "x1": max(k, 2) + 1,
            "y1": y,
            "line": {"color": color, "width": width, "dash": dash},
        }

    def label(y, text):
        return {
            "x": 0.75,
            "y": y,
            "xref": "paper",
            "yref": "y",
            "text": text,
            "showarrow": False,
            "font": {"color": "white"},
        }

    axis_font = {"color": "darkgray"}
    layout = dict(
        margin=dict(t=40),
        hovermode="closest",
        uirevision=col,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.15},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "zeroline": False,
            "showgrid": False,
            "title": {"text": "Batch Number", "font": axis_font},
            "showline": False,
            "domain": [0, 0.8],
        },
        yaxis={
            "title": {"text": col, "font": axis_font},
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
        },
        annotations=[
            label(lim["lcl"], f"LCL: {lim['lcl']:.4g}"),
            label(lim["ucl"], f"UCL: {lim['ucl']:.4g}"),
            label(lim["usl"], f"USL: {lim['usl']:.4g}"),
            label(lim["lsl"], f"LSL: {lim['lsl']:.4g}"),
            label(lim["mean"], f"Baseline mean: {lim['mean']:.4g}"),
        ],
        shapes=[
            hline(lim["usl"], "#91dfd2"),
            hline(lim["lsl"], "#91dfd2"),
            hline(lim["ucl"], "rgb(255,127,80)"),
            hline(lim["lcl"], "rgb(255,127,80)"),
            hline(lim["mean"], "rgb(255,127,80)", width=2, dash="solid"),
        ],
        xaxis2={
            "title": {"text": "Count", "font": axis_font},
            "domain": [0.8, 1],
            "showgrid": False,
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
        },
    )
    return {"data": traces, "layout": layout}


def capability_figure(interval, store):
    k = clamp_k(interval)
    caps = [capability(m, store, k) for m in metrics]
    cpk = [c["cpk"] for c in caps]
    colors = [
        "#92e0d3" if v >= CPK_TARGET else "#f4d44d" if v >= 1 else "#f45060" for v in cpk
    ]
    hover = [
        f"{m}<br>Cp {c['cp']:.2f} | Cpk {c['cpk']:.2f} | Ppk {c['ppk']:.2f}"
        f"<br>{c['out_spec']:.1f}% outside spec"
        for m, c in zip(metrics, caps)
    ]
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": cpk,
                "y": metrics,
                "marker": {"color": colors},
                "text": [f"{v:.2f}" for v in cpk],
                "textposition": "outside",
                "hovertext": hover,
                "hoverinfo": "text",
            }
        ],
        "layout": {
            "margin": dict(l=110, r=30, t=20, b=40),
            "font": {"color": "darkgray"},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "title": {"text": "Cpk (green = 1.33 or more)", "font": {"color": "darkgray"}},
                "showgrid": False,
                "zeroline": False,
                "range": [min(0, min(cpk) - 0.2), max(2, max(cpk) + 0.3)],
            },
            "yaxis": {"autorange": "reversed"},
            "shapes": [
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": CPK_TARGET,
                    "x1": CPK_TARGET,
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
                }
            ],
            "uirevision": True,
        },
    }


# --------------------------------------------------------------------------------------
# App layout and callbacks
# --------------------------------------------------------------------------------------
app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=TICK_MS,
            n_intervals=START_N,
            max_intervals=max_length,  # simulation ends when all batches are shown
            disabled=False,
        ),
        html.Div(
            id="app-container",
            children=[build_tabs(), html.Div(id="app-content")],
        ),
        dcc.Store(id="value-setter-store", data=initial_store()),
        dcc.Store(id="n-interval-stage", data=START_N),
        dcc.Store(id="selected-metric", data=metrics[0]),
        generate_modal(),
        html.Div(
            id="footer",
            children=[
                "Demonstration dataset bundled with Plotly's sample app; illustrative only, not real "
                "production data. Adapted from Plotly's open-source dash-manufacture-spc-dashboard "
                "(MIT licence). Extended by Pawel Sogra: moving-range control limits, run rules, "
                "capability (Cp/Cpk/Ppk), exception log, working specification settings."
            ],
        ),
    ],
)


@app.callback(
    [Output("app-content", "children"), Output("interval-component", "n_intervals")],
    [Input("app-tabs", "value")],
    [State("n-interval-stage", "data")],
)
def render_tab_content(tab_switch, stopped_interval):
    if tab_switch == "tab1":
        return build_tab_1(), stopped_interval
    return (
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=[build_top_panel(stopped_interval), build_chart_panel()],
                ),
            ],
        ),
        stopped_interval,
    )


@app.callback(
    Output("n-interval-stage", "data"),
    [Input("app-tabs", "value")],
    [
        State("interval-component", "n_intervals"),
        State("interval-component", "disabled"),
        State("n-interval-stage", "data"),
    ],
)
def update_interval_state(tab_switch, cur_interval, disabled, cur_stage):
    # remember how far the feed got while the user is on the settings tab
    if tab_switch == "tab1":
        return cur_interval
    return cur_stage


@app.callback(
    [Output("interval-component", "disabled"), Output("stop-button", "buttonText")],
    [Input("stop-button", "n_clicks")],
    [State("interval-component", "disabled")],
)
def stop_production(n_clicks, current):
    if not n_clicks:
        return False, "stop"  # feed starts running on load
    return not current, "stop" if current else "start"


@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context
    if ctx.triggered:
        if ctx.triggered[0]["prop_id"].split(".")[0] == "learn-more-button":
            return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("progress-gauge", "value"), [Input("interval-component", "n_intervals")]
)
def update_gauge(interval):
    return clamp_k(interval)


@app.callback(
    [Output("alerts-led", "value"), Output("alerts-led", "color")],
    [Input("interval-component", "n_intervals")],
    [State("value-setter-store", "data")],
)
def update_alerts_led(interval, store):
    k = clamp_k(interval)
    total = 0
    for m in metrics:
        _, any_flag, _ = analyse(m, store)
        total += int(any_flag[max(0, k - 20) : k].sum())
    color = "#92e0d3" if total == 0 else "#f4d44d" if total < 5 else "#f45060"
    return str(min(total, 99)), color


# ---------------- Specification Settings tab ----------------
SPEC_ROWS = [
    ("usl", "Upper Specification limit"),
    ("lsl", "Lower Specification limit"),
    ("ucl", "Upper Control limit"),
    ("lcl", "Lower Control limit"),
]


def build_value_setter_rows():
    """Static rows (inputs always exist); a callback fills in baselines and placeholders."""
    rows = [
        build_value_setter_line(
            "value-setter-panel-header", "Specs", "Baseline value", "Set new value"
        )
    ]
    for key, label in SPEC_ROWS:
        rows.append(
            build_value_setter_line(
                f"value-setter-panel-{key}",
                label,
                html.Span(id=f"baseline-{key}"),
                dcc.Input(
                    id=f"ud_{key}_input",
                    type="number",
                    className="setting-input",
                    debounce=True,
                ),
            )
        )
    return rows


@app.callback(
    [Output(f"baseline-{k}", "children") for k, _ in SPEC_ROWS]
    + [Output(f"ud_{k}_input", "placeholder") for k, _ in SPEC_ROWS]
    + [Output(f"ud_{k}_input", "value") for k, _ in SPEC_ROWS],
    [Input("metric-select-dropdown", "value"), Input("value-setter-store", "data")],
)
def fill_value_setter_panel(dd_select, store):
    base = BASE_LIMITS[dd_select]
    cur = store[dd_select]
    return (
        [f"{base[k]:.4g}" for k, _ in SPEC_ROWS]
        + [f"{cur[k]:.4g}" for k, _ in SPEC_ROWS]
        + [None for _ in SPEC_ROWS]
    )


@app.callback(
    [Output("value-setter-store", "data"), Output("value-setter-msg", "children")],
    [
        Input("value-setter-set-btn", "n_clicks"),
        Input("value-setter-reset-btn", "n_clicks"),
    ],
    [
        State("metric-select-dropdown", "value"),
        State("value-setter-store", "data"),
        State("ud_usl_input", "value"),
        State("ud_lsl_input", "value"),
        State("ud_ucl_input", "value"),
        State("ud_lcl_input", "value"),
    ],
    prevent_initial_call=True,
)
def set_value_setter_store(set_btn, reset_btn, param, data, usl, lsl, ucl, lcl):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    if trigger == "value-setter-reset-btn":
        data[param] = _round_limits(BASE_LIMITS[param])
        return data, f"{param}: reset to baseline limits."

    new = dict(data[param])
    for key, val in (("usl", usl), ("lsl", lsl), ("ucl", ucl), ("lcl", lcl)):
        if val is not None:
            new[key] = float(val)
    if new["usl"] <= new["lsl"]:
        return dash.no_update, "Upper specification limit must be above the lower limit."
    if new["ucl"] <= new["lcl"]:
        return dash.no_update, "Upper control limit must be above the lower limit."
    data[param] = new
    return data, f"{param}: limits updated."


@app.callback(
    Output("value-setter-view-output", "children"),
    [
        Input("value-setter-view-btn", "n_clicks"),
        Input("metric-select-dropdown", "value"),
        Input("value-setter-store", "data"),
    ],
)
def show_current_specs(n_clicks, dd_select, store_data):
    if not n_clicks:
        return None
    cur = store_data[dd_select]
    base = BASE_LIMITS[dd_select]
    rows = [
        ("Upper Specification Limit", "usl"),
        ("Lower Specification Limit", "lsl"),
        ("Upper Control Limit", "ucl"),
        ("Lower Control Limit", "lcl"),
    ]
    data = [
        {"Specs": name, "Baseline": f"{base[k]:.4g}", "Current Setup": f"{cur[k]:.4g}"}
        for name, k in rows
    ]
    return dash_table.DataTable(
        style_header={"fontWeight": "bold", "color": "inherit"},
        style_as_list_view=True,
        fill_width=True,
        style_cell_conditional=[{"if": {"column_id": "Specs"}, "textAlign": "left"}],
        style_cell={
            "backgroundColor": "#1e2130",
            "fontFamily": "Open Sans",
            "padding": "0 2rem",
            "color": "darkgray",
            "border": "none",
        },
        css=[
            {"selector": "tr:hover td", "rule": "color: #91dfd2 !important;"},
            {"selector": "td", "rule": "border: none !important;"},
            {
                "selector": ".dash-cell.focused",
                "rule": "background-color: #1e2130 !important;",
            },
            {"selector": "table", "rule": "--accent: #1e2130;"},
            {"selector": "tr", "rule": "background-color: transparent"},
        ],
        data=data,
        columns=[{"id": c, "name": c} for c in ["Specs", "Baseline", "Current Setup"]],
    )


# ---------------- Dashboard tab: per-parameter rows ----------------
def make_row_callback(param):
    def callback(interval, store):
        k = clamp_k(interval)
        _, _, ooc_cum = analyse(param, store)
        ooc_f = float(ooc_cum[k - 1] * 100)
        grad = 0.00001 if ooc_f == 0 else min(ooc_f, 15.0)
        if grad <= 5:
            color = "#92e0d3"
        elif grad < ALARM_PCT:
            color = "#f4d44d"
        else:
            color = "#FF0000"
        return str(k), sparkline_figure(param, k), f"{ooc_f:.2f}%", grad, color

    return callback


for _param in metrics:
    app.callback(
        output=[
            Output(_param + suffix_count, "children"),
            Output(_param + suffix_sparkline_graph, "figure"),
            Output(_param + suffix_ooc_n, "children"),
            Output(_param + suffix_ooc_g, "value"),
            Output(_param + suffix_indicator, "color"),
        ],
        inputs=[Input("interval-component", "n_intervals")],
        state=[State("value-setter-store", "data")],
    )(make_row_callback(_param))


@app.callback(
    Output("selected-metric", "data"),
    [Input(m + suffix_button_id, "n_clicks") for m in metrics],
    prevent_initial_call=True,
)
def select_metric(*clicks):
    ctx = dash.callback_context
    for trig in ctx.triggered:
        prop_id = trig["prop_id"].split(".")[0]
        if prop_id.endswith(suffix_button_id) and trig["value"]:
            return prop_id[: -len(suffix_button_id)]
    return dash.no_update


@app.callback(
    Output("control-chart-live", "figure"),
    [
        Input("interval-component", "n_intervals"),
        Input("selected-metric", "data"),
        Input("value-setter-store", "data"),
    ],
)
def update_control_chart(interval, metric, store):
    return generate_graph(interval, store, metric if metric in metrics else metrics[0])


@app.callback(
    Output("capability-chart", "figure"),
    [Input("interval-component", "n_intervals"), Input("value-setter-store", "data")],
)
def update_capability(interval, store):
    return capability_figure(interval, store)


@app.callback(
    Output("alert-log", "children"),
    [Input("interval-component", "n_intervals"), Input("value-setter-store", "data")],
)
def update_alert_log(interval, store):
    k = clamp_k(interval)
    events = []
    for m in metrics:
        flags, any_flag, _ = analyse(m, store)
        for i in np.flatnonzero(any_flag[:k])[-8:]:
            rule = next(r for r in ("r1", "r2", "r4") if flags[r][i])
            events.append((int(i), m, rule, float(df[m].iloc[i])))
    events.sort(key=lambda e: e[0], reverse=True)
    events = events[:8]
    if not events:
        return html.P("No alarms so far.", className="alert-empty")
    return html.Table(
        className="alert-table",
        children=[
            html.Thead(
                html.Tr([html.Th(h) for h in ("Batch", "Parameter", "Rule", "Value")])
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(batch_x[i]),
                            html.Td(m),
                            html.Td(RULE_LABELS[r]),
                            html.Td(f"{v:.5g}"),
                        ]
                    )
                    for i, m, r, v in events
                ]
            ),
        ],
    )


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
        port=int(os.environ.get("PORT", 8050)),
    )
