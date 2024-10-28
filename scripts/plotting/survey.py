import marimo

__generated_with = "0.8.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell
def __(pl):
    results = pl.read_csv("artifacts/survey.csv")
    return (results,)


@app.cell
def __(results):
    results
    return


@app.cell
def __(alt, results):
    _chart = (
        alt.Chart(
            results,
            title="Survey respondents by role",
        )
        .transform_aggregate(count="count()", groupby=["role"])
        .transform_filter(alt.datum.count > 1)
        .mark_bar()
        .encode(
            x=alt.X(
                "role:N",
                scale=alt.Scale(
                    domain=[
                        "Undergraduate Student",
                        "Masters Student",
                        "Ph.D. Student",
                        "Postdoc",
                        "Professor",
                    ]
                ),
                title="",
            ),
            y=alt.Y("count:Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=24, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/roles.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = results["areas"].str.split(", ").explode().to_frame()

    _chart = (
        alt.Chart(
            _results,
            title="Research Areas of Survey Respondents",
        )
        .mark_bar()
        .transform_aggregate(count="count()", groupby=["areas"])
        .transform_filter(alt.datum.count > 2)
        .encode(
            x=alt.X("areas:N", title=""),
            y=alt.Y("count:Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=24, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/areas.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = (
        results["uses"]
        .str.split(", ")
        .explode()
        .replace(
            {
                "Training (with small data / models)": "Training",
                "Pre-training (with large data / models)": "Pre-training",
            }
        )
        .to_frame()
    )

    _chart = (
        alt.Chart(
            _results,
            title="Use Cases of Survey Respondents",
        )
        .mark_bar()
        .transform_aggregate(count="count()", groupby=["uses"])
        .transform_filter(alt.datum.count > 2)
        .encode(x=alt.X("uses:N"), y=alt.Y("count:Q"))
    )

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _chart = (
        alt.Chart(
            results,
            title=alt.Title("Satisfaction with Available Compute", subtitle="(5 = most satisfied)"),
        )
        .mark_bar()
        .encode(
            x=alt.X("satisfaction:N", axis=alt.Axis(labelAngle=0), title=""),
            y=alt.Y("count():Q", title="Respondents"),
        )
        .properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=24, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/satisfaction.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _chart = (
        alt.Chart(
            results,
            title="Cloud Compute Budgets of Survey Respondents",
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "cloud_budget:Q",
                bin=alt.Bin(step=500),
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y("count():Q"),
        )
    )

    _chart.interactive()
    return


@app.cell
def __(mo):
    mo.md(r"""## GPU Generations""")
    return


@app.cell
def __(alt, results):
    _results = (
        results["desktop_gen"]
        .str.split(", ")
        .explode()
        .map_elements(lambda x: x.split(" ")[0], return_dtype=str)
        .to_frame()
    )

    _chart = (
        alt.Chart(
            _results,
            title="Deskop GPUs by Generation",
        )
        .transform_filter(
            (alt.datum.desktop_gen != "")
            & (alt.datum.desktop_gen != "Not")
            & (alt.datum.desktop_gen != "None")
            & alt.expr.isValid(alt.datum.desktop_gen)
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "desktop_gen:N",
                scale=alt.Scale(
                    domain=[
                        "Pascal",
                        "Volta",
                        "Turing",
                        "Ampere",
                        "Lovelace",
                        "[AMD]",
                    ]
                ),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/desktop_gen.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = (
        results["workstation_gen"]
        .str.split(", ")
        .explode()
        .map_elements(lambda x: x.split(" ")[0], return_dtype=str)
        .to_frame()
    )

    _chart = (
        alt.Chart(
            _results,
            title="Workstation GPUs by Generation",
        )
        .transform_filter(
            (alt.datum.workstation_gen != "")
            & (alt.datum.workstation_gen != "Not")
            & (alt.datum.workstation_gen != "None")
            & alt.expr.isValid(alt.datum.workstation_gen)
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "workstation_gen:N",
                scale=alt.Scale(
                    domain=[
                        "Turing",
                        "Ampere",
                        "Lovelace",
                        "[AMD]",
                    ]
                ),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/workstation_gen.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = (
        results["data_center_gen"]
        .str.split(", ")
        .explode()
        .map_elements(lambda x: x.split(" ")[0], return_dtype=str)
        .to_frame()
    )

    _chart = (
        alt.Chart(
            _results,
            title="Data Center GPUs by Generation",
        )
        .transform_filter(
            (alt.datum.data_center_gen != "")
            & (alt.datum.data_center_gen != "Not")
            & (alt.datum.data_center_gen != "None")
            & alt.expr.isValid(alt.datum.data_center_gen)
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "data_center_gen:N",
                scale=alt.Scale(
                    domain=[
                        "Pascal",
                        "Volta",
                        "Turing",
                        "Ampere",
                        "Hopper",
                        "Lovelace",
                        "[AMD]",
                    ]
                ),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/data_center_gen.pdf")

    _chart.interactive()
    return


@app.cell
def __(mo):
    mo.md(r"""## GPU Memory""")
    return


@app.cell
def __(alt, results):
    _results = results["desktop_mem"].str.split(", ").explode().to_frame()

    _chart = (
        alt.Chart(
            _results,
            title="Deskop GPUs by Memory",
        )
        .transform_filter(
            (alt.datum.desktop_mem != "Not sure")
            & (alt.datum.desktop_mem != "None available")
            & alt.expr.isValid(alt.datum.desktop_mem)
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "desktop_mem:N",
                scale=alt.Scale(
                    domain=[
                        "4 GB",
                        "8 GB",
                        "12 GB",
                        "16 GB",
                        "20 GB",
                        "24 GB",
                        "32 GB",
                    ]
                ),
                axis=alt.Axis(labelAngle=0),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/desktop_mem.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = results["workstation_mem"].str.split(", ").explode().to_frame()

    _chart = (
        alt.Chart(
            _results,
            title="Workstation GPUs by Memory",
        )
        .transform_filter(
            (alt.datum.workstation_mem != "Not sure")
            & (alt.datum.workstation_mem != "None available")
            & alt.expr.isValid(alt.datum.workstation_mem)
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "workstation_mem:N",
                scale=alt.Scale(domain=["16 GB", "24 GB", "32 GB", "48 GB"]),
                axis=alt.Axis(labelAngle=0),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents"),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/workstation_mem.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = results["data_center_mem"].str.split(", ").explode().to_frame()

    _chart = (
        alt.Chart(
            _results,
            title="Data Center GPUs by Memory",
        )
        .transform_filter(
            (alt.datum.data_center_mem != "Not sure")
            & (alt.datum.data_center_mem != "None available")
            & alt.expr.isValid(alt.datum.data_center_mem)
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "data_center_mem:N",
                scale=alt.Scale(domain=["32 GB", "48 GB", "64 GB", "80 GB"]),
                axis=alt.Axis(labelAngle=0),
                title="",
            ),
            y=alt.Y("count():Q", title=""),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/data_center_mem.pdf")

    _chart.interactive()
    return


@app.cell
def __(mo):
    mo.md(r"""## GPU Usage Durations""")
    return


@app.cell
def __(pl, results):
    def process_gpu_durations(prefix):
        _num_gpus = ["1", "2", "4", "8", "16", "32", "64"]

        _results = results.rename({f"{prefix}_{_c}": _c for _c in _num_gpus}).select(
            _num_gpus + [f"{prefix}_gen", f"{prefix}_mem"]
        )

        # If "gen" or "mem" columns equal "None available"
        # Set all values to "N/A"

        _results = _results.select(
            [
                pl.when((pl.col(f"{prefix}_gen") == "None available") | (pl.col(f"{prefix}_mem") == "None available"))
                .then(pl.lit("N/A"))
                .otherwise(pl.col(col))
                .alias(col)
                for col in _num_gpus
            ]
        ).drop_nulls()

        # To: [num_gpus, time, count]

        _results = (
            _results.unpivot(on=_num_gpus, variable_name="num_gpus", value_name="")
            .to_dummies("", separator="")
            .group_by("num_gpus")
            .sum()
            .unpivot(index="num_gpus", variable_name="time", value_name="count")
        )

        _total_counts = _results.group_by("num_gpus").sum().drop("time")
        _results = _results.join(_total_counts, on=["num_gpus"], suffix="_total")
        _results = _results.with_columns(count_normalized=(pl.col("count") / pl.col("count_total")))

        _results = _results.filter(pl.col("time") != "N/A")

        # Writing: "64+" instead of "64" GPUs
        _results = _results.with_columns(
            num_gpus=pl.when(pl.col("num_gpus") == "64").then(pl.lit("64+")).otherwise(pl.col("num_gpus")),
        )

        return _results

    return (process_gpu_durations,)


@app.cell
def __(alt, process_gpu_durations):
    def build_gpu_avail_chart(type):
        _results = process_gpu_durations(type)

        _chart = (
            (
                alt.Chart(_results)
                .transform_calculate(
                    time_order="{'Hours': 0, 'Days': 1, 'Weeks': 2, 'Months': 3, 'Indefinitely': 4}[datum.time]"
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "num_gpus",
                        sort=["0", "1", "2", "4", "8", "16", "32", "64+"],
                        title="Number of GPUs",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("count_normalized", title="Respondents", axis=alt.Axis(tickCount=5, format="%")),
                    color=alt.Color(
                        "time",
                        sort=alt.SortField("time_order", "ascending"),
                        title="",
                        legend=alt.Legend(labelFontSize=22, orient="top", columns=3, legendX=-100),
                    ),
                    order=alt.Order("time_order:O", sort="descending"),
                )
            )
            .properties(
                height=400,
                width=400,
                title=alt.TitleParams(
                    text=f"Availability of {' '.join(map(str.capitalize, type.split('_')))} GPUs",
                    fontSize=28,
                    anchor="middle",
                ),
            )
            .configure_axis(labelFontSize=22, titleFontSize=28)
        )

        return _chart

    return (build_gpu_avail_chart,)


@app.cell
def __(build_gpu_avail_chart):
    _chart = build_gpu_avail_chart("desktop")

    _chart.save("artifacts/plots/survey/desktop_availability.pdf")

    _chart.interactive()
    return


@app.cell
def __(build_gpu_avail_chart):
    _chart = build_gpu_avail_chart("workstation")

    _chart.save("artifacts/plots/survey/workstation_availability.pdf")

    _chart.interactive()
    return


@app.cell
def __(build_gpu_avail_chart):
    _chart = build_gpu_avail_chart("data_center")

    _chart.save("artifacts/plots/survey/data_center_availability.pdf")

    _chart.interactive()
    return


@app.cell
def __(mo):
    mo.md(r"""## Inter-GPU/Node Connections""")
    return


@app.cell
def __(alt, results):
    _results = results["gpu_link"].str.split(", ").explode().to_frame()

    _chart = (
        alt.Chart(
            _results,
            title="Inter-GPU Connections",
        )
        .transform_filter((alt.datum.gpu_link != "Not sure") & alt.expr.isValid(alt.datum.gpu_link))
        .mark_bar()
        .encode(
            x=alt.X(
                "gpu_link:N",
                scale=alt.Scale(domain=["No", "NVLink", "NVSwitch", "AMD Infinity Fabric"]),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents", axis=alt.Axis(tickCount=5)),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/gpu_link.pdf")

    _chart.interactive()
    return


@app.cell
def __(alt, results):
    _results = (
        results["node_link"]
        .replace(
            {
                "No multi-node support": "None",
                "Multi-node supported (but connectivity is unknown)": None,
                "Not sure": None,
            }
        )
        .map_elements(lambda x: x.split(" ")[0], return_dtype=str)
        .to_frame()
    )

    _chart = (
        alt.Chart(
            _results,
            title="Inter-Node Connections",
        )
        .transform_filter(alt.expr.isValid(alt.datum.node_link))
        .mark_bar()
        .encode(
            x=alt.X(
                "node_link:N",
                scale=alt.Scale(
                    domain=[
                        "None",
                        "Ethernet",
                        "Infiniband",
                    ]
                ),
                title="",
            ),
            y=alt.Y("count():Q", title="Respondents", axis=alt.Axis(tickCount=5)),
        )
    )

    _chart = (
        _chart.properties(width=400, height=400)
        .configure_title(fontSize=24, subtitleFontSize=20)
        .configure_axis(labelFontSize=20, titleFontSize=24)
    )

    _chart.save("artifacts/plots/survey/node_link.pdf")

    _chart.interactive()
    return


if __name__ == "__main__":
    app.run()
