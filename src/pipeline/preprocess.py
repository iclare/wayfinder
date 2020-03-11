import sys
from pathlib import Path

import pandas as pd

from tqdm import tqdm

from src.io.read import get_placemarks, read_gps_to_coordinates
from src.io.serialize import cached
from src.io.write import write_coordinates_to_kml
from src.pipeline.util import row_in_region


def _angle_delta(a1, a2):
    """Get the difference in angle between two angles.
    :param a1: First angle in degrees.
    :param a2: Second angle in degrees.
    :return: The angle difference in degrees, positive or negative.
    """
    r = (a2 - a1) % 360.0
    if r >= 180.0:
        r -= 360.0
    return r


def _enroute_coords(coords, spd):
    """Determine which coordinates are after departure and before arrival.
    :param coords: DataFrame of coordinates.
    :param spd: Speed to consider the first and last coordinates of the route.
    :return: The coordinates DataFrame without pre departure and post arrival coordinates.
    """
    over_spd = coords.index[coords.spd_over_grnd > spd].tolist()
    if len(over_spd) == 0:
        # Indicate that these coordinates are to be ignored since the route never exceeds the
        # minimum speed for departure
        return None
    coords = coords[coords.index > over_spd[0]] if len(over_spd) > 0 else coords
    coords = coords[coords.index < over_spd[-1]] if len(over_spd) > 0 else coords
    return coords


def _row_in_pmarks(row, pmarks):
    """Return whether this row's coordinates are contained within the placemark.
    :param row: DataFrame row with lat and lon coordinates.
    :param pmarks: The placemarks to check.
    :return: True or False.
    """
    # Return the new label for the record if it is in the place mark rectangle.
    for pmark in pmarks:
        in_placemark_rectangle = row_in_region(row, pmark)
        if in_placemark_rectangle:
            return True
    return False


def _split_two_way_route(coords, will_score, pmarks_path):
    """If this route goes to b and back a, then split the route into two different routes, one
     where it goes to b and one where it goes a. Coordinates near a or b are filtered out.
    :param coords: DataFrame with coords of the route.
    :param will_score: Whether this route will be scored. Routes that won't be scored can still be
        used for training.
    :param pmarks_path: Path containing the pmarks of regions (b or a).
    :return: Tuple of coordinate DataFrames (to_b, to_a), one of which may be None.
    """
    # Make sure the route starts and ends at a or b
    a_pmarks = get_placemarks(Path(f"{pmarks_path}/a.kml"))
    b_pmarks = get_placemarks(Path(f"{pmarks_path}/b.kml"))
    coords["a"] = coords.apply(
        lambda row: _row_in_pmarks(row, a_pmarks),
        axis=1
    )
    coords["b"] = coords.apply(
        lambda row: _row_in_pmarks(row, b_pmarks),
        axis=1
    )
    # Check that the start and end location of the route are at a or b
    if (
            (not coords.iloc[-1].b and not coords.iloc[-1].a)
            or (not coords.iloc[0].b and not coords.iloc[0].a)
    ):
        # Indicate the start or end location of the route is invalid
        return None

    # If the start region is the same as the end region, this file must include a route starting at
    # a and b. Split the file into two different routes.
    to_b, to_a = None, None
    try:
        if coords.iloc[0].b and coords.iloc[-1].b and will_score:
            # Must have gone a and back. Split the data when parked at a.
            stopped_at_a_idx = coords[coords.a].datetime_delta.idxmax()
            to_a = coords[coords.index <= stopped_at_a_idx]
            to_b = coords[coords.index > stopped_at_a_idx]
        elif coords.iloc[0].a and coords.iloc[-1].a and will_score:
            # Must have gone to b and back. Split the data when parked at b.
            stopped_at_b_idx = coords[coords.b].datetime_delta.idxmax()
            to_b = coords[coords.index <= stopped_at_b_idx]
            to_a = coords[coords.index > stopped_at_b_idx]
    except ValueError as e:
        # Indicates that the route could not be split into valid start and end areas. This happens
        # when the route does not contain coords at b or a.
        return None
    # If we have a two way route that we are splitting, we need to make sure the to a and to b
    # routes only start above some min mph again. Ignore data until we depart and after we arrive.
    if to_b is not None and to_a is not None:
        to_b = to_b.drop(["b", "a"], axis=1)
        to_a = to_a.drop(["b", "a"], axis=1)
        return to_b, to_a
    else:
        # Determine which position the coords belongs to in the tuple indicating whether this route
        # is going to b or going a. (`b`, `a`). If we started at a, we are going to
        # b. If we started at b, we are going a.
        is_to_b = False
        is_to_a = False
        if coords.iloc[0].a:
            is_to_b = True
        elif coords.iloc[0].b:
            is_to_a = True
        coords = coords.drop(["b", "a"], axis=1)
        if is_to_b:
            return coords, None
        if is_to_a:
            return None, coords


def _calc_feats(coords, delta_shifts, mean_neighbors):
    """Calculate features that help identify coordinates that are a part of a turn or stop.
    :param coords: Coords DataFrame to create new columns for.
    :param delta_shifts: How many coordinates ahead in the route deltas are computed for.
    :param mean_neighbors: How large should the window be for the rolling mean.
    :return: The DataFrame with rolling mean and rolling mean delta columns.
    """
    for n_nearest in mean_neighbors:
        coords[f"true_course_rolling_mean{n_nearest}"] = coords.true_course.rolling(
            n_nearest, win_type="triang", center=True
        ).mean()
        coords[f"spd_over_grnd_rolling_mean{n_nearest}"] = coords.spd_over_grnd.rolling(
            n_nearest, win_type="triang", center=True
        ).mean()
        for shift in delta_shifts:
            coords[f"true_course_rolling_mean{n_nearest}_delta{shift}"] = (
                coords.join(coords.shift(shift), rsuffix="_next").apply(
                    lambda row: _angle_delta(
                        a1=getattr(row, f"true_course_rolling_mean{n_nearest}"),
                        a2=getattr(row, f"true_course_rolling_mean{n_nearest}_next")
                    ),
                    axis=1
                )
            )
            coords[f"spd_over_grnd_rolling_mean{n_nearest}_delta{shift}"] = (
                (getattr(coords, f"spd_over_grnd_rolling_mean{n_nearest}")
                 - getattr(coords, f"spd_over_grnd_rolling_mean{n_nearest}").shift(shift))
            )

        # Remove unused column
        coords = coords.drop(f"true_course_rolling_mean{n_nearest}", axis=1)
    return coords


def _preprocess_coords(
        coords, will_score,
        region_pmarks_path,
        arrive_depart_speed=15,
):
    """Preprocess coordinates to filter noise, remove outliers, ignore coordinates outside of the
    valid area, ignore routes that are used for scoring that do not visit both a and b, split
    two way routes into one way routes, and create columns that will be used for training,
    (moving averages, deltas)
    :param coords: DataFrame containing raw coordinate data for the route.
    :param will_score: Whether this route will be used for scoring.
    :param region_pmarks_path: Path where placemarks are stored indicating b or a regions.
    :param arrive_depart_speed: Minimum speed to consider for first and last coords of the route.
    :param filter_strength: How much to filter data using filtfilt.
    :return: A DataFrame of the coordinates after preprocessing.
    """
    # Remove coordinates where we are stopped.
    # This speed needs to be high enough to remove drifting
    # coordinates but low enough that we can consider it as being stopped.
    coords[f"spd_over_grnd_rolling_mean16"] = coords.spd_over_grnd.rolling(
        16, win_type="triang", center=True
    ).mean()
    coords = coords[coords.spd_over_grnd_rolling_mean16 > 0.1]
    coords = coords.drop("spd_over_grnd_rolling_mean16", axis=1)

    # Ignore data until we depart and after we arrive. This means after we first hit min mph and
    # before the last time we go below min mph.
    coords = _enroute_coords(coords, arrive_depart_speed)
    if coords is None:
        print("\nEncountered route that never departs", file=sys.stderr)
        return None  # We never departed

    # Remove invalid coordinates
    valid_area_placemarks = get_placemarks(Path(f"{region_pmarks_path}/valid_area.kml"))
    coords["in_valid_area"] = coords.apply(
        lambda row: _row_in_pmarks(row, valid_area_placemarks),
        axis=1
    )
    coords = coords[coords.in_valid_area].drop("in_valid_area", axis=1)

    # Create columns that are the difference between speed, speed delta, true course delta for each
    # row compared to a previous row. Calculate rolling means for each.
    coords = _calc_feats(
        coords,
        delta_shifts=(1, 16),
        mean_neighbors=(2, 8)
    )

    # Drop rows with nulls
    coords = coords.dropna()

    # Remove drifting points at a low speed.
    coords = coords[~((coords.spd_over_grnd_rolling_mean2 <= 1.0)
                      & (coords.true_course_rolling_mean2_delta1.abs() >= 45.0))]
    # Remove noisy angle deltas
    coords = coords[coords.true_course_rolling_mean2_delta1.abs() <= 75.0]

    # Account for time gaps in the data (store the seconds difference to the next row)
    coords["datetime_delta"] = (coords.datetime - coords.datetime.shift(1)).dt.total_seconds()

    # If this route's destination is the same as it's origin, split it into multiple routes.
    # If this route will be scored, make sure the route contains at least a and b.
    coords_start_end = _split_two_way_route(
        coords,
        will_score=will_score,
        pmarks_path=region_pmarks_path
    )
    if coords_start_end is None:
        # Indicate that the coordinates do not
        # start and end in valid areas
        print("\nCould not split route into valid start and end areas", file=sys.stderr)
        return None
    # This route either goes to b or a, or both.
    to_b, to_a = coords_start_end
    if to_b is not None:
        to_b = to_b.dropna()
    if to_a is not None:
        to_a = to_a.dropna()
    return to_b, to_a


def _coords_to_records(coords, attribs):
    """Return a list of records represented by a dict.
    :param coords: List of RMC objects.
    :param attribs: Attributes to keep.
    :return: List of dict holding the attributes we are interested for coordinates.
    """
    records = []
    for coord in coords:
        record = {}
        try:
            for attrib in attribs:
                record[attrib] = getattr(coord, attrib)
            records.append(record)
        except TypeError as e:
            # Ignore the rare RMC object that
            # was parsed incorrectly by pynmea
            continue
    return records


def _raw_coords(gps_file):
    """Return a DataFrame containing coordinate data parsed from the given gps file.
    :param gps_file: gps file containing NMEA sentences.
    :return: DataFrame with records of nmea sentence data.
    """
    coords = read_gps_to_coordinates(gps_file)
    coords = _coords_to_records(
        coords,
        attribs=["latitude", "longitude", "spd_over_grnd", "datetime", "true_course"]
    )
    coords = pd.DataFrame(coords)
    coords["spd_over_grnd"] *= 1.15078  # knots to mph
    return coords


def _create_data(files, kind, out_path, region_pmarks_path, kml_out_dest_label=True):
    """Create the preprocessed DataFrame for the routes of these files.
    :param files: The files to process.
    :param kind: What kind of routes we are processing (unseen, training)
    :param out_path: Where to store the preprocessed data.
    :param kml_out_dest_label: Whether to add `.<destination>` before `.kml`
    :return: DataFrame with the preprocessed coordinate data.
    """
    dataframes = []
    files_used = []
    for file in tqdm(files, desc=f"Preprocessing {kind} routes"):
        coordinates = _raw_coords(file)
        write_coordinates_to_kml(Path(f"{out_path / file.name}.raw.kml"), coordinates)
        coordinates = _preprocess_coords(coordinates,
                                         will_score=kind == "unseen",
                                         region_pmarks_path=region_pmarks_path)
        if coordinates is None:
            print(f"Route will not be used: {file.name}\n", file=sys.stderr)
            continue
        for i, data in enumerate(coordinates):
            if data is None:
                continue  # No to b or to a route
            fname_label_to_b = ".to_b" if kml_out_dest_label else ""
            fname_label_to_a = ".to_a" if kml_out_dest_label else ""
            oneway_file = (Path(f"{file.name}{fname_label_to_b}") if i == 0 else
                           Path(f"{file.name}{fname_label_to_a}"))
            write_coordinates_to_kml(Path(f"{out_path / oneway_file.name}.preproc.kml"), data)
            dataframes.append(data)
            files_used.append(oneway_file)
    return dataframes, files_used


@cached("unseen_data")
def create_unseen_data(unseen_files, unseen_out_path, region_pmarks_path):
    """Process routes as was done for the training files but these files route coordinates are not
    labeled. Routes that cannot be split into either or both routes that go a and to b are not
    processed.
    :param unseen_files: The files without labels for coordinates.
    :param unseen_out_path: Where to read the routes.
    :param region_pmarks_path: Where to read placemarks indicating b, a, and valid route area.
    :return: Tuple containing a list of coordinate DataFrames and a list of unseen files that were
        deemed usable.
    """
    return _create_data(
        unseen_files,
        kind="unseen",
        out_path=unseen_out_path,
        region_pmarks_path=region_pmarks_path
    )


def _label_regions(coordinates, file, label_pmarks_path, classif_label):
    """Label any coordinates with the given label found in the given placemark.
    :param coordinates: Coordinates DataFrame.
    :param file: Name of the file for the route being processed. Used to get associated placemarks.
    :param label_pmarks_path: Where the placemark files containing labeled regions is found.
    :param classif_label: What to label the coordinates found in the placemark.
    :return: The coordinates DataFrame with labels.
    """
    pmarks = get_placemarks(Path(f"{label_pmarks_path / file.name}.{classif_label}.kml"))
    coordinates.label = coordinates.apply(
        lambda row: classif_label if _row_in_pmarks(row, pmarks) else row.label,
        axis=1
    )
    return coordinates


def _training_data_for_coords(coordinates, file, out_path, label_pmarks_path, labels):
    """Return the coordinates DataFrame with all labels applied.
    :param coordinates: Coordinates DataFrame.
    :param file: Name of file for the route.
    :param out_path: Where to write the kml for the route before labeling.
    :param label_pmarks_path: Where to read labels from.
    :param labels: The possible labels.
    :return: The coordinates DataFrame with all possible labels.
    """
    write_coordinates_to_kml(Path(f"{out_path / file.name}.preproc.kml"), coordinates)
    # Label regions of the training data
    for label in labels:
        coordinates = _label_regions(
            coordinates,
            file,
            label_pmarks_path,
            label
        )
    return coordinates


@cached("training_data")
def create_training_data(training_files, training_out_path, region_pmarks_path, label_pmarks_path):
    """Preprocess, split, reject and label coordinates for routes given for training. Routes that do
     not depart and do not start at b or a are rejected.
    :param training_files: Files to process.
    :param training_out_path: Where to store intermediate kml files.
    :param region_pmarks_path: Where to read placemarks identifying b a, and valid regions.
    :param label_pmarks_path: Where to read placemarks that identify the label for training
        coordinates.
    :return: A tuple containing list of coordinate DataFrames, and list of files that were deemed
        usable associated by index.
    """
    # Parse, preprocess, and validate the data to use for training.
    training_dataframes, training_files_used = _create_data(
        training_files,
        kind="training",
        out_path=training_out_path,
        region_pmarks_path=region_pmarks_path,
        kml_out_dest_label=False
    )

    # Label coordinates in a labeled region placemark.
    labeled_dataframes = []
    for dataframe, file in zip(training_dataframes, training_files_used):
        labeled_dataframe = _training_data_for_coords(
            dataframe,
            file,
            out_path=training_out_path,
            label_pmarks_path=label_pmarks_path,
            labels=["stopped", "turning_left", "turning_right"]
        )
        labeled_dataframes.append(labeled_dataframe)

    return labeled_dataframes, training_files_used
