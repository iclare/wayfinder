

from geopy import distance as gpdistance


def row_in_region(row, region):
    """Return whether the coordinates of this row is within the rectangular region.
    :param row: Pandas DataFrame row with coordinate lon and lat.
    :param region: Rectangular area to check.
    :return: True or False
    """
    # coord[0] - lon
    # coord[1] - lat
    bottom_left, top_left, bottom_right, top_right = region
    # is to right left edge and is to left of right edge
    in_horizontal = (
            (row.longitude > top_left[0] and row.longitude > bottom_left[0])
            and (row.longitude < top_right[0] and row.longitude < bottom_right[0])
    )
    # is below top edge and is above bottom edge
    in_vertical = (
            (row.latitude < top_left[1] and row.latitude < top_right[1])
            and (row.latitude > bottom_left[1] and row.latitude > bottom_right[1])
    )
    in_placemark_rectangle = in_vertical and in_horizontal
    return in_placemark_rectangle


def distance(coordinate1, coordinate2):
    """Compute the Vincenty distance between two coordinates.
    :param coordinate1: tuple representing a coordinate (lon, lat)
    :param coordinate2: tuple representing a coordinate (lon, lat)
    :return: The distance between two coordinates in feet.
    """
    c1_lat_lon = coordinate1[1], coordinate1[0]
    c2_lat_lon = coordinate2[1], coordinate2[0]
    return gpdistance.geodesic(c1_lat_lon, c2_lat_lon).feet


def compute_distance_deltas(rows, shift=-1):
    """Compute the distance deltas between coordinates in a DataFrame.
    :param rows: The DataFrame to process.
    :param shift: Direction of the delta. -1 For x(t1) - x(t0).
    :return: A Series of distance deltas.
    """
    return rows.join(rows.shift(shift).dropna(), rsuffix="_next").dropna().apply(
        lambda row: distance(
            (row.longitude, row.latitude),
            (row.longitude_next, row.latitude_next)
        ),
        axis=1
    )
