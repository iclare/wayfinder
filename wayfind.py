"""Score routes to A or B using a cost function that primarily penalizes travel
time, and then penalizes gas usage and tire wear to choose between routes with
similar travel times.

The cost function:
    cost              = objective + regularization
    objective         = travel_time(min) / 30(min)
    regularization    = (1/10)(max_velocity_cost + turns_cost + stoppage_cost)

    max_velocity_cost = (max_velocity(mph) / 60(mph)
    turns_cost        = (n_left_turns / 5) + (1/2)(n_right_turns / 5)
    stoppage_cost     = (n_stop_lights / 5) + (1/2)(n_stop_signs / 5)

    Where typical route values are:
        max_velocity: 60(mph)
        n_left_turns, n_right_turns: 5, 5
        n_stop_lights, n_stop_signs: 5, 5

Traffic lights cost twice as much as stop signs, as they are associated with
high traffic areas and high speed areas, contributing more to gas and tire wear.

Left turns cost twice as much as right turns, as they are associated with large
periods of waiting for an opening at an intersection.

Stops and turns are identified by training a gradient boosted tree on hand
inspected and labeled training routes that have been preprocessed. Training
routes contain coordinates that are labeled as being part of either a stop, a
left turn, a right turn, or neither.

Unseen route coordinates are preprocessed and then labeled. Routes are then
prepared for scoring, scored, and a recommended route is given for going to A or
to B.

The first time this program is ran, preprocessed coordinates and the model for
classification will be computed and serialized. Subsequent runs will reuse the
serialized data.

See README.md for parameters to recompute preprocessing or training.
"""

__author__ = "Ian Clare"
__date__ = "01/12/2020"

import sys

from pathlib import Path
import pandas as pd

import lightgbm as lgb
import matplotlib.pyplot as plt

from src.io.write import write_coordinates_summary, write_routes_kml
from src.pipeline.classify import classify_unseen_data
from src.pipeline.preprocess import create_training_data, create_unseen_data
from src.pipeline.train import fit_model
from src.pipeline.score import score_routes


def parse_args():
    """Parse arguments given. When no arguments are given, all serialized data will be reused if
    found. If serialized data is not found, the data is computed.
    Keyword arguments:
        --invalidate=preprocessing: Preprocessed and training model data will be recomputed.
        --invalidate=training: Training model data will be recomputed and serialized.
    :return: The parsed arguments as a tuple.
    """
    # Defaults
    invalidate_preprocessing = False
    invalidate_training = False

    # Use defaults for no arguments.
    if not len(sys.argv) > 1:
        return invalidate_preprocessing, invalidate_training

    # Only expecting '--invalidate=<val>' with <val> as either 'preprocessing' or 'training'
    if not sys.argv[1].startswith("--invalidate="):
        sys.exit(f"Unrecognized argument: {sys.argv[1]}")
    val = sys.argv[1].split("=")[1]
    if val not in ("preprocessing", "training", "scoring"):
        sys.exit(f"Unrecognized value in: {sys.argv[1]}")

    # Set argument dependents.
    invalidate_preprocessing = val == "preprocessing"
    # Always invalidate training when we invalidate preprocessing.
    invalidate_training = invalidate_preprocessing or val == "training"

    return invalidate_preprocessing, invalidate_training


def collapsed(dict_, sep="_"):
    """Flatten a nested dictionary.

    >>> collapsed({
    ...    'a': {'b': 0},
    ...    'c': {
    ...        'd': 1,
    ...        'e': {'f': 2, 'g': 3}
    ...    }
    ... })
    {'a_b': 0, 'c_d': 1, 'c_e_f': 2, 'c_e_g': 3}
    >>> collapsed({'a': {}})
    {}
    >>> collapsed({})
    {}
    """
    new = {}
    for k, v in dict_.items():
        if type(v) != dict:
            new[k] = v
            continue
        for k_, v_ in collapsed(v, sep).items():
            new[k_] = v_
    return new


def create_scored_routes_table(scored_routes):
    """Prep route for summary and analysis by creating a DataFrame."""
    route_datas = []
    for route in scored_routes:
        route_data = {}
        for k, v in collapsed(route).items():
            if isinstance(v, Path):
                assert "to_b" in v.name or "to_a" in v.name
                route_data["destination"] = "B" if "to_b" in v.name else "A"
                v = f"{v.name}.scored.kml"
            elif isinstance(v, pd.DataFrame):
                continue
            route_data[k] = v
        route_datas.append(route_data)
    scored_routes_table = pd.DataFrame(route_datas)
    return scored_routes_table.set_index("file")


def main():
    # Whether we should invalidate preprocessing and or training data. When invalidated, data will
    # not be deserialized and instead will be recomputed.
    invalidate_preprocessing, invalidate_training = parse_args()
    # Placemark paths.
    region_pmarks_path = Path(f"./in/placemarks/")
    label_pmarks_path = Path(f"./in/placemarks/training")

    ##
    # Train
    ##
    # Preprocessing for training data.
    training_out_path = Path(f"./out/training/")
    training_files = list(Path().glob("./in/training/*"))
    training_sets, training_files = create_training_data(
        training_files=training_files,
        training_out_path=training_out_path,
        region_pmarks_path=region_pmarks_path,
        label_pmarks_path=label_pmarks_path,
        invalidate=invalidate_preprocessing
    )
    # Write out the labeled kml for inspection.
    write_routes_kml(
        routes=training_sets,
        files=training_files,
        out_path=training_out_path,
        file_label="labeled"
    )
    # Features to use for training. Any delta or rolling_mean column.
    # Keep columns that have 'delta' or 'rolling_mean' in the name.
    features = [col for col in training_sets[0].columns
                if any([substr in col for substr in ["delta", "rolling_mean"]])]
    # Train the model.
    fitted_model, label_encoder, evals_result = fit_model(
        training_sets=training_sets,
        features=features,
        invalidate=invalidate_training
    )
    # Inspect training if it was recomputed.
    if invalidate_training:
        # Save plot of the training metric.
        training_img = Path("./analysis/training.png")
        print(f"\nPlotting metrics during training and saving to {training_img.name}.")
        lgb.plot_metric(evals_result, metric="multi_logloss")
        plt.show()
        # Save plot of feature importances.
        feature_importances_img = Path("./analysis/feature_importances.png")
        print(f"\nPlotting feature importances and saving to {feature_importances_img.name}.\n")
        lgb.plot_importance(fitted_model, max_num_features=len(features))
        plt.show()

    ##
    # Classify
    ##
    # Preprocessing for unseen data.
    unseen_out_path = Path(f"./out/unseen/")
    unseen_files = list(Path().glob("./in/unseen/*"))
    unseen_sets, unseen_files = create_unseen_data(
        unseen_files=unseen_files,
        unseen_out_path=unseen_out_path,
        region_pmarks_path=region_pmarks_path,
        invalidate=invalidate_preprocessing
    )
    # Classify the unseen data.
    classified_unseen_data = classify_unseen_data(
        model=fitted_model,
        unseen_sets=unseen_sets,
        features=features,
        label_encoder=label_encoder,
        invalidate=invalidate_training
    )
    # Write out the classified kml for inspection.
    write_routes_kml(
        routes=classified_unseen_data,
        files=unseen_files,
        out_path=unseen_out_path,
        file_label="classified"
    )

    ##
    # Score Routes
    ##
    # Make a recommendation for a route to take when going to A or to B.
    # Routes that do not start at either B or A are ignored.
    scored_routes = score_routes(
        classified_unseen_data=classified_unseen_data,
        files=unseen_files,
        invalidate=False
    )
    # Sort the routes by cost ascending.
    scored_routes = sorted(scored_routes, key=lambda data: data["cost"])
    # Write out the scored routes
    # Separate the routes by their destination.
    scored_to_b_routes = [rdata for rdata in scored_routes if "to_b" in rdata["file"].name]
    scored_to_a_routes = [rdata for rdata in scored_routes if "to_a" in rdata["file"].name]
    # Write out the scored routes.
    scored_out_path = Path("./out/scored_unseen/")
    write_routes_kml(
        routes=[data["route"] for data in scored_to_b_routes],
        files=[data["file"] for data in scored_to_b_routes],
        out_path=scored_out_path,
        file_label="scored",
        stops_as_path=False,
        turns_as_path=False,
        altitude_as_speed=False
    )
    write_routes_kml(
        routes=[data["route"] for data in scored_to_a_routes],
        files=[data["file"] for data in scored_to_a_routes],
        out_path=scored_out_path,
        file_label="scored",
        stops_as_path=False,
        turns_as_path=False,
        altitude_as_speed=False
    )

    ##
    # Summarize
    ##
    # Training routes KML summary.
    write_coordinates_summary(
        kml_filename=Path("./out/training_routes_summary.kml"),
        coords_list=training_sets,
        fnames=training_files
    )
    # Unseen routes KML summary.
    write_coordinates_summary(
        kml_filename=Path("./out/unseen_routes_summary.kml"),
        coords_list=unseen_sets,
        fnames=unseen_files
    )
    # All routes KML summary.
    write_coordinates_summary(
        kml_filename=Path("./out/all_routes_summary.kml"),
        coords_list=training_sets + unseen_sets,
        fnames=training_files + unseen_files
    )
    # Scored to B routes KML summary.
    write_coordinates_summary(
        kml_filename=Path("./out/scored_to_b_summary.kml"),
        coords_list=[data["route"] for data in scored_to_b_routes],
        fnames=[data["file"] for data in scored_to_b_routes]
    )
    # Scored to A routes KML summary.
    write_coordinates_summary(
        kml_filename=Path("./out/scored_to_a_summary.kml"),
        coords_list=[data["route"] for data in scored_to_a_routes],
        fnames=[data["file"] for data in scored_to_a_routes]
    )
    # Summarize scores.
    pd.set_option("display.max_colwidth", -1)
    pd.set_option("display.expand_frame_repr", False)
    routes_table = create_scored_routes_table(scored_routes)
    to_b_table = routes_table[routes_table.destination == "B"].drop("destination", axis=1)
    to_a_table = routes_table[routes_table.destination == "A"].drop("destination", axis=1)
    # Show the route scores for going to B.
    print("###### To B Scored Routes ######")
    print(to_b_table)
    # Show the route scored for going A.
    print("###### To A Scored Routes ######")
    print(to_a_table)
    # Show the route scores..
    print("###### All Scored Routes  ######")
    print(routes_table)
    print()


if __name__ == "__main__":
    main()
