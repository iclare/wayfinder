from collections import defaultdict
from pathlib import Path
from string import Template

_style_poly_format = Template("""\
<Style id="${id_}">
  <LineStyle>
    <color>${color}</color>
    <width>6</width>
  </LineStyle>
  <PolyStyle>
    <color>${color}</color>
  </PolyStyle>
</Style>
""")

_path_placemark_format = Template("""\
<Placemark>
<styleUrl>#${style_id}</styleUrl>
<name>${description}</name>
<LineString>
<Description>${description}</Description>
  <extrude>1</extrude>
  <tesselate>1</tesselate>
  <altitudeMode>relativeToGround</altitudeMode>
  <coordinates>
${coords}
  </coordinates>
</LineString>
</Placemark>
""")

_coordinate_placemark_format = Template("""\
<Placemark>
<description>${description}</description>
<styleUrl>#${style_id}</styleUrl>
<name>${description}</name>
<Point>
  <coordinates>${coord}</coordinates> 
</Point>
</Placemark>
""")

_document_format = Template(Template("""\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>

${style_green_poly}
${style_red_poly}
${style_blue_poly}
${style_cyan_poly}
${style_yellow_poly}


${default_placemark}
${stopped_placemark}
${turning_left_placemarks}
${turning_right_placemarks}
${stop_light_placemarks}
${stop_sign_placemarks}

</Document>
</kml>
""").safe_substitute(
    style_green_poly=_style_poly_format.substitute(id_="greenPoly", color="af00ff00"),
    style_red_poly=_style_poly_format.substitute(id_="redPoly", color="af0000ff"),
    style_blue_poly=_style_poly_format.substitute(id_="bluePoly", color="afff0000"),
    style_cyan_poly=_style_poly_format.substitute(id_="cyanPoly", color="afffff00"),
    style_yellow_poly=_style_poly_format.substitute(id_="yellowPoly", color="af00c4f5")
))


def _format_coord(coord, altitude_as_speed=True, baseline_alt=10):
    # Each coordinate needs to be in the format (longitude, latitude, altitude).
    # Speed is added to the altitude column for visualization.
    latitude = "{0:.6f}".format(coord.latitude)
    longitude = "{0:.6f}".format(coord.longitude)
    altitude = "{0:.1f}".format(
        (coord.spd_over_grnd + baseline_alt) if altitude_as_speed  # Visualize speed
        else baseline_alt
    )
    coord_str = f"{longitude},{latitude},{altitude}"
    return coord_str


def _format_coordinates(coords, altitude_as_speed, **kwargs):
    """Format the coordinate data for kml.
    :param coords: Coordinate DataFrame with latitude, longitude, and spd_over_grnd columns.
    :return: A string with each coordinate of the DataFrame on a new line as longitude, latitude,
        spd_over_grnd.
    """
    coord_strs = []
    for coord in coords.itertuples():
        coord_str = _format_coord(coord, altitude_as_speed, **kwargs)
        coord_strs.append(coord_str)
    return "\n".join(coord_strs)


def _format_coord_pmark(style_id, description, coords):
    """Create multiple individual placemarks for single coordinates."""
    coord_strs = list(  # Format the coordinates. Don't show speed for single placemarks.
        coords.apply(lambda row: _format_coord(row, altitude_as_speed=False), axis=1)
    ) if len(coords) > 0 else []

    return "\n".join(
        _coordinate_placemark_format.substitute(
            style_id=style_id,
            description=description,
            coord=coord_str
        ) for coord_str in coord_strs
    )


def write_coordinates_to_kml(
        kml_filename,
        coords,
        stops_as_path=True,
        turns_as_path=True,
        altitude_as_speed=True
):
    """Write the DataFrame of coordinate data to kml file for viewing in Google Earth. Left turns
    will be in blue, stops in red, right turns in cyan, everything else in green.
    :param kml_filename: The file to write to.
    :param coords: DataFrame containing coordinate data.
    :param stops_as_path: Whether to show actual coordinates considered as part of a stop or a pmark
    :param turns_as_path: Whether to show the actual coordinates considered as turning or one pmark.
    :param altitude_as_speed: Whether to vary path latitude by speed to visualize speed.
    """

    def _path_or_placemarks(style_id, description, coords_, as_path):
        # Whether we are writing turns and stops as paths or single placemarks.
        if as_path:
            return _path_placemark_format.substitute(
                style_id="bluePoly",
                description="Turning Left",
                coords=_format_coordinates(coords_, altitude_as_speed)
            )
        return _format_coord_pmark(
            style_id=style_id,
            description=description,
            coords=coords_
        )

    # Set record labels to default if there there is no label column.
    if "label" not in coords.columns:
        coords["label"] = "default"
    # Make sure no rows contain null values.
    coords = coords.dropna()

    with open(kml_filename, "w+") as kml_file:
        kml_out = _document_format.substitute(
            default_placemark=_path_placemark_format.substitute(
                style_id="greenPoly",
                description="Default",
                coords=_format_coordinates(coords[coords.label == "default"], altitude_as_speed)
            ),
            stopped_placemark=_path_placemark_format.substitute(
                style_id="redPoly",
                description="stopped",
                coords=_format_coordinates(coords[coords.label == "stopped"], altitude_as_speed)
            ),
            turning_left_placemarks=_path_or_placemarks(
                style_id="bluePoly",
                description="Turning Left",
                coords_=coords[coords.label == "turning_left"],
                as_path=turns_as_path
            ),
            turning_right_placemarks=_path_or_placemarks(
                style_id="cyanPoly",
                description="Turning Right",
                coords_=coords[coords.label == "turning_right"],
                as_path=turns_as_path
            ),
            stop_light_placemarks=_path_or_placemarks(
                style_id="redPoly",
                description="Stop Light",
                coords_=coords[coords.label == "stop_light"],
                as_path=stops_as_path
            ),
            stop_sign_placemarks=_path_or_placemarks(
                style_id="redPoly",
                description="Stop Sign",
                coords_=coords[coords.label == "stop_sign"],
                as_path=stops_as_path
            )
        )
        kml_file.write(kml_out)


def write_routes_kml(
        routes,
        files,
        out_path,
        file_label,
        stops_as_path=True,
        turns_as_path=True,
        altitude_as_speed=True
):
    """Write route data with labels to separate kml files.
    :param routes: The route DataFrames to write.
    :param files: The files of the classified routes.
    :param out_path: Were to write the kml files.
    :param file_label: Label for the file.
    :param stops_as_path: Whether to show actual coordinates considered as part of a stop or a pmark
    :param turns_as_path: Whether to show the actual coordinates considered as turning or one pmark.
    :param altitude_as_speed: Whether to vary path latitude by speed to visualize speed.
    """
    for classified_unseen_data, file in zip(routes, files):  # Iterate in pairs.
        write_coordinates_to_kml(
            Path(f"{out_path / file.name}.{file_label}.kml"),
            classified_unseen_data,
            stops_as_path,
            turns_as_path,
            altitude_as_speed
        )


def write_coordinates_summary(kml_filename, coords_list, fnames):
    """Create a summary kml file of the coordinates in the routes given. The kml file will show
    all routes given as separate paths.
    :param kml_filename: What to name the kml file.
    :param coords_list: List of coordinate dataframes to write.
    :param fnames: The file names to be used as the description for routes.
    """
    empty_str_defaultdict = defaultdict(lambda: "")
    fmt = _format_coordinates
    route_strs = (
        _path_placemark_format.substitute(
            style_id="greenPoly",
            description=fname.name,
            coords=fmt(coords, altitude_as_speed=True)
        ) for (coords, fname) in zip(coords_list, fnames)
    )
    empty_str_defaultdict["default_placemark"] = "\n".join(route_strs)
    kml_out = _document_format.substitute(empty_str_defaultdict)
    with open(kml_filename, "w+") as kml_file:
        kml_file.write(kml_out)
