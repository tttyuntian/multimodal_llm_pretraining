import polars as pl


def process_training_time_results(
    results: pl.DataFrame,
    free_lunch_only: bool = False,
    mem_saving_only: bool = False,
    select_min: bool = False,
):
    if free_lunch_only:
        results = results.filter(
            pl.col("activation_checkpointing") == False, pl.col("sharding") == "", pl.col("offloading") == False
        )
    elif mem_saving_only:
        results = results.filter(
            (
                (pl.col("activation_checkpointing") != False)
                | (pl.col("sharding") != "")
                | (pl.col("offloading") != False)
            )
        )

    if select_min:
        _group_columns = ["num_nodes", "gpus_per_node", "gpu_type", "model"]
        _results_min = results.group_by(_group_columns).agg(pl.col("training_days").min())

        results = results.join(_results_min, on=_group_columns + ["training_days"], how="inner")

    return results
