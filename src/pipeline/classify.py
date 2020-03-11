from src.io.serialize import cached


@cached("classified_unseen_data")
def classify_unseen_data(model, unseen_sets, features, label_encoder):
    """Return the data for unseen routes with coordinates classified.
    :param model: Model to classify with
    :param unseen_sets: List of unseen route data.
    :param features: Features to use for prediction
    :param label_encoder: Label encoder used to decode the label name.
    :return: List of routes with coordinates classified
    """
    classified_unseen_data = []
    for unseen_set in unseen_sets:
        feats = unseen_set[features]
        prediction = model.predict(feats)
        unseen_set["label"] = label_encoder.inverse_transform(prediction.argmax(axis=1))
        classified_unseen_data.append(unseen_set)
    return classified_unseen_data
