import sys

import pynmea2
from fastkml import kml
from pynmea2 import ParseError, RMC


def read_gps_to_coordinates(gps_filename):
    """Read the file containing NMEA sentences into a list of pyneam2 RMC objects.
    :param gps_filename: The file to read.
    :return: List of RMC objects.
    """
    msgs = []
    with open(gps_filename) as gps_file:
        for line in gps_file:
            try:
                msg = pynmea2.parse(line, check=True)
            except ParseError as e:
                continue
            if isinstance(msg, RMC):
                msgs.append(msg)
    return msgs


def get_placemarks(path_to_file):
    """Read a placemarks file representing a rectangular region and return the coordinates in order.
    :param path_to_file: Path to the file containing the region of placemarks.
    :return: List of Tuple of (longitude, latitude) tuples in order by position on the globe of
        earth with the pole straight up. Ordered as (bottom_left, top_left, bottom_right, top_right)
        Each tuple of coordinate tuple in the list represents a rectangular region.
    """
    try:
        doc = path_to_file.read_bytes()
    except FileNotFoundError:
        print(f"\nPlacemark file not found: {path_to_file} ", file=sys.stderr)
        sys.exit(1)

    # Extract the placemarks of each region.
    k = kml.KML()
    k.from_string(doc)
    k.etree_element()
    features = list(k.features())
    f2 = list(features[0].features())

    # Reorder the placemark / coordinates of each rectangular region by position
    # (bottom_left, top_left, bottom_right, top_right).
    regions = []
    for placemark in f2:
        geoms = placemark.geometry.exterior.geoms
        c1 = geoms[0].bounds[:2]
        c2 = geoms[1].bounds[:2]
        c3 = geoms[2].bounds[:2]
        c4 = geoms[3].bounds[:2]
        coords = c1, c2, c3, c4
        sorted_by_lon = sorted(coords, key=lambda x: x[0])
        bottom_left, top_left = sorted(sorted_by_lon[:2], key=lambda x: x[1])
        bottom_right, top_right = sorted(sorted_by_lon[2:], key=lambda x: x[1])
        coords = bottom_left, top_left, bottom_right, top_right
        regions.append(coords)
    return regions
