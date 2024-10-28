import polars as pl

if __name__ == "__main__":
    pl.read_csv(
        "artifacts/raw_survey.csv",
        columns=list(range(4, 38)),
        new_columns=[
            "role",
            "areas",
            "uses",
            "satisfaction",
            "cloud_budget",
            "desktop_gen",
            "desktop_mem",
            "desktop_1",
            "desktop_2",
            "desktop_4",
            "desktop_8",
            "desktop_16",
            "desktop_32",
            "desktop_64",
            "workstation_gen",
            "workstation_mem",
            "workstation_1",
            "workstation_2",
            "workstation_4",
            "workstation_8",
            "workstation_16",
            "workstation_32",
            "workstation_64",
            "data_center_gen",
            "data_center_mem",
            "data_center_1",
            "data_center_2",
            "data_center_4",
            "data_center_8",
            "data_center_16",
            "data_center_32",
            "data_center_64",
            "gpu_link",
            "node_link",
        ],
    ).write_csv("artifacts/survey.csv")