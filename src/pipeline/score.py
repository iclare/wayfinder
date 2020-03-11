import sys
from concurrent import futures
from multiprocessing import cpu_count

from tqdm import tqdm

from src.pipeline.util import compute_distance_deltas, distance
from src.io.serialize import cached


def _collapse_label(coords, label, region_size_ft):
    """Collapse a particular labeled coordinate to one within a region and recompute time deltas.
    :param coords: Coordinate DataFrame of a route.
    :param label: The label to collapse in regions.
    :param region_size_ft: The size of the region to collapse.
    :return: The collapsed coordinates DataFrame.
    """
    # This will keep the last coordinate part of the stop region to indicate a stop.
    delete_coords = coords[coords.label == label].copy()
    if len(delete_coords) > 1:
        delete_coords["distance_delta"] = compute_distance_deltas(delete_coords)
        # | is the `or` operator in pandas
        delete_coords = delete_coords[
            (delete_coords.distance_delta < region_size_ft)
            | (delete_coords.distance_delta.isnull())  # Catch last region
        ]
        coords = coords.drop(delete_coords.index)
    # Recompute the datetime_delta.
    coords["datetime_delta"] = (
            coords.datetime.shift(-1) - coords.datetime
    ).shift(1).dt.total_seconds()
    return coords


def _collapsed_labels_and_stop_regions(
        classified_unseen_data,
        region_size_ft,
        stop_sign_median_secs_min=1,
        stop_light_median_secs_min=10,
        parked_secs=60 * 15
):
    """Determine where there are stop signs and stop lights across all the classified routes.
    :param classified_unseen_data: DataFrames with coordinate data of a route.
    :param region_size_ft: How large of a region (in ft) should stop areas be collapsed.
    :return: Tuple of DataFrames containing (stop_signs, stop_lights, stop_collapsed data).
    """
    stop_regions = None
    for i, coords in tqdm(  # Progress bar
            enumerate(classified_unseen_data),
            desc="Postprocessing",
            total=len(classified_unseen_data)
    ):
        # Collapse labels in close proximity, keeping the newer label. This will keep the last
        # coordinate part of the stop region to indicate a stop.
        for label in ["stopped", "turning_left", "turning_right"]:
            coords = _collapse_label(coords, label, region_size_ft)

        # We are not interested in parked data.
        coords = coords[coords.datetime_delta < parked_secs].dropna()
        # Update the original route with collapsed labels.
        classified_unseen_data[i] = coords

        # Track stop regions
        stops = coords[coords.label == "stopped"].copy()
        if stop_regions is None:
            stop_regions = stops.reset_index(drop=True)
        else:
            stop_regions = stop_regions.append(stops, ignore_index=True, sort=False)

    # Determine stop region statistics and choose a prototype coordinate.
    stop_regions = stop_regions.sort_values(by=["latitude", "longitude"]).reset_index(drop=True)
    stop_regions["distance_delta"] = compute_distance_deltas(stop_regions)
    stop_regions["region_id"] = -1
    last_group_idxs = stop_regions[
        (stop_regions.distance_delta > region_size_ft)
        | (stop_regions.distance_delta.isnull())  # Catch the last region
        ].index.to_list()
    for region_id, idx in enumerate(last_group_idxs):
        # & is the `and` operator in pandas
        selected_rows = (stop_regions.index <= idx) & (stop_regions.region_id == -1)
        stop_regions.loc[selected_rows, "region_id"] = region_id
    kept_columns = stop_regions.columns.difference(["latitude", "longitude",
                                                    "datetime_delta", "region_id"])
    stop_regions = stop_regions.drop(kept_columns, axis=1).groupby(["region_id"]).median()

    def region_type(row):
        if row.datetime_delta < stop_sign_median_secs_min:
            return "traffic"
        if row.datetime_delta < stop_light_median_secs_min:
            return "stop_sign"
        return "stop_light"

    stop_regions["type"] = stop_regions.apply(region_type, axis=1)
    # We won't try to determine high traffic regions as this is not part of the route cost function.
    stop_regions = stop_regions.drop(stop_regions[stop_regions.type == "traffic"].index)
    stop_regions = stop_regions.dropna()
    return stop_regions, classified_unseen_data


def _route_cost(
        travel_time_mins,
        max_velocity_mph,
        n_left_turns,
        n_right_turns,
        n_stop_signs,
        n_stop_lights,
        travel_time_norm=30,
        max_velocity_norm=60,
        n_left_turns_norm=5,
        n_right_turns_norm=5,
        n_stop_signs_norm=5,
        n_stop_lights_norm=5
):
    """Penalize the route based on time and vehicle energy expenditure. The route will be penalized
    mainly based on the time to arrival, and then based on any factors contributing to tire wear and
    gas usage.

    * Penalize stop signs as they waste time, gas and tire life.
    * Stop lights are penalized even more because of the increased time and gas usage.
    * We penalize turns because of the tire and gas usage associated with them.
    * Because left turns usually cost more time than right turns,
      we penalize it more.
    """
    left_turns_cost = n_left_turns / n_left_turns_norm
    right_turns_cost = (1 / 2) * (n_right_turns / n_right_turns_norm)

    n_stop_lights_cost = n_stop_lights / n_stop_lights_norm
    n_stop_signs_cost = (1 / 2) * (n_stop_signs / n_stop_signs_norm)

    # Regularization components
    max_velocity_cost = max_velocity_mph / max_velocity_norm
    turns_cost = left_turns_cost + right_turns_cost
    stoppage_cost = n_stop_lights_cost + n_stop_signs_cost

    # We care mostly about the time taken to arrive.
    objective = travel_time_mins / travel_time_norm
    # Regularization to help choose between two routes with similar arrival times.
    regularization = (1 / 10) * (max_velocity_cost + turns_cost + stoppage_cost)

    cost = objective + regularization

    components = dict(
        cost=cost,
        regularization=regularization,
        objective=objective,
        max_velocity_cost=max_velocity_cost,
        stoppage_cost=stoppage_cost,
        turns_cost=turns_cost
    )
    return components


def _score_route(
        classified_route_data,
        stop_regions,
        region_size_ft,
        file,
        max_stop_sign_mean_spd=10,
        max_gap_size_ft=200
):
    """Calculate the cost of a route as the score. A Smaller score / cost is better.
    :param classified_route_data: DataFrame containing coordinate data for the route.
    :param stop_regions: Regions to check against for stops.
    :param region_size_ft: How close a coordinate has to get to a kind of stop before it counts.
    :param max_stop_sign_mean_spd: Max rolling avg speed to consider a coord as passing a stop sign.
        This prevents labeling a coordinate as passing a stop sign when the stop sign is at a
        different side from the vehicle's side of the junction.
    :param max_gap_size: Max size of a gap in the data before removing it from consideration.
    :return: The score of the route and number of gaps in the route.
    """
    # Don't score the route if it contains a large gap.
    classified_route_data["distance_delta"] = compute_distance_deltas(
        classified_route_data
    ).dropna()
    if (classified_route_data.distance_delta > max_gap_size_ft).any():
        print("\nEncountered impossible movement in unseen route", file=sys.stderr)
        return None

    def assign_stop_regions(stop_row):
        # Assigns labels to rows that are the closest to a stop region.
        # Allowing us to determine whether this route passes a previously identified
        # stop sign or stop light.
        min_ = classified_route_data.apply(
            lambda coord_row: distance((coord_row.longitude, coord_row.latitude),
                                       (stop_row.longitude, stop_row.latitude)),
            axis=1
        )
        min_dist = min_.min()
        if min_dist > region_size_ft:
            return  # Did not pass the previously seen stop.
        min_idx = min_.idxmin()
        if stop_row.type == "stop_sign":
            correct_side = (classified_route_data.loc[min_idx]
                            .spd_over_grnd_rolling_mean8 < max_stop_sign_mean_spd)
            if not correct_side:
                return  # The stop sign was on the wrong side.
        # Assign the label.
        classified_route_data.loc[min_idx, "label"] = stop_row.type

    # Stop labels were used to detect stop signs and stop lights. They are no longer needed.
    classified_route_data.loc[classified_route_data.label == "stopped", "label"] = "default"
    stop_regions.apply(assign_stop_regions, axis=1)
    label_counts = classified_route_data["label"].value_counts()

    travel_time_mins = classified_route_data.datetime_delta.sum() / 60,  # s to min
    travel_time_mins = int(travel_time_mins[0] if type(travel_time_mins == tuple)
                           else travel_time_mins)  # This is a tuple sometimes.
    max_velocity_mph = int(classified_route_data.spd_over_grnd.max())
    n_left_turns = label_counts.get("turning_left", 0)  # Zero if label does not exist in route.
    n_right_turns = label_counts.get("turning_right", 0)
    n_stop_signs = label_counts.get("stop_sign", 0)
    n_stop_lights = label_counts.get("stop_light", 0)

    stats = dict(
        travel_time_mins=travel_time_mins,
        max_velocity_mph=max_velocity_mph,
        n_left_turns=n_left_turns,
        n_right_turns=n_right_turns,
        n_stop_signs=n_stop_signs,
        n_stop_lights=n_stop_lights,
    )
    cost_components = _route_cost(**stats)
    return dict(cost=cost_components.pop("cost"),
                cost_components=cost_components,
                stats=stats,
                file=file,
                route=classified_route_data)


@cached("route_scores")
def score_routes(classified_unseen_data, files, region_size_ft=60):
    """Score routes based on how many times that route has passed through a stop sign, how many
    times that route stopped through a stop light, the number of left turns, the number of right
    turns, and the time elapsed since departure and arrival.
    :param classified_unseen_data: List of data for each route with coordinates classified as either
        being part of a stop, a right turn, or a left turn.
    :param files: The files associated with the data.
    :param region_size_ft: How far to look to collapse labels into a single point and to consider
        a coordinate as having passed a previously seen stop light or stop sign. If this is set too
        high we will think the route has a stop sign. This created a radius around a coordinate
        prototype stop light or stop sign coordinate.
    :return: Dataframe with scored routes data.
    """
    # First we need to know the regions where there is a stop sign and a stop light.
    stop_regions, classified_unseen_data = _collapsed_labels_and_stop_regions(
        classified_unseen_data,
        region_size_ft
    )

    # Vincenty distance is expensive to compute for every previous stop sign and stop light we've
    # seen. So utilize however many cores are available. This is computed to determine whether the
    # route passes a stop sign or stop light even for routes where a light is green, or a stop sign
    # is ran. Results are not returned in order, so track the associated file.
    with futures.ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        results = []

        for classified_route_data, file in zip(classified_unseen_data, files):
            res = pool.submit(
                # function
                _score_route,
                # args
                classified_route_data,
                stop_regions,
                region_size_ft,
                file
            )
            results.append(res)

        results = list(tqdm(
            (res.result() for res in futures.as_completed(results)),
            desc="Scoring routes",
            total=len(results)
        ))
        returned = []
        for result in results:
            if result is None:
                print(f"\nRoute will not be scored {file.name}. A gap was found.", file=sys.stderr)
            else:
                returned.append(result)

        return returned

