# wayfinder

Finds the best routes between two points of interest.

The basic idea is to:

1. Label example route data. Sections of the route are labeled as `stopped`, `turned left` or `turned right`.
2. Pre-process all route data to cleanup noise, and detect unrecoverable faults.
3. Drop and create features to form clusters of records with similar probability of being any one label.
3. Estimate ideal parameters for training using hyper parameter optimization and cross validation.
4. Train a model on the labeled route data using the found parameters, and gradient tree boosting.
5. Classify sections of the routes in consideration with our labels.
6. Utilize classifications between overlapping route sections to detect traffic instruments like a stop sign / light.
7. Report route statistics and favorability.

## Viewing KML

KML files can be viewed in [Google Earth](https://www.google.com/earth/)

## Quick-Start

Example files are given for the files mentioned below.
Simply delete or replace them with your own files.
KML files can be created using Google Earth.

* Add a `in/placemarks/a.kml` containing a rectangular placemark surrounding one of your locations.
* Add a `in/placemarks/b.kml` containing a rectangular placemark surround your other location.
* Add NMEA format files to `in/training/` to be used for training.
* Add NMEA format files to `in/unseen/` to be used for classification.
* See *Running* to run the program.
* Now using the files `out/training/*.raw.kml`.
  * Open each file in Google Earth and produce the kml files containing rectangular regions corresponding to a label as seen in `in/placemarks/training`
* See *Running to run the program again. A report will be displayed showing which routes were favorable.

## Running

```
pip3 install pipenv
pipenv install
pipenv run python wayfind.py
```

View intermediate and scored files in `./out/`.

## Optional Arguments

Recompute and serialize preprocessing and training data:
```
pipenv run python wayfind.py --invalidate=preprocessing
```

Recompute and serialize training data:
```
pipenv run python wayfind.py --invalidate=training
```

