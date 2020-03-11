from pprint import pprint

import pandas as pd
import lightgbm as lgb
import numpy as np

from joblib import cpu_count
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe

from src.io.serialize import cached


def _prepare_train_set(sets, features, label_encoder):
    """Collapse multiple data sets to a single LightGBM data set with encoded labels, and features
    for training.
    """
    data = pd.concat(sets, ignore_index=True)[features + ["label"]]
    labels = label_encoder.fit_transform(data.label)
    data = data.drop("label", axis=1)
    dataset = lgb.Dataset(data, labels)
    return dataset


@cached("model")
def fit_model(training_sets, features, seed=50):
    """LightGBM is used which is a gradient boosted tree. LightGBM's trees are grown leaf wise
    rather than level wise, and data is binned by default for a performance advantage, but is more
    susceptible to over fitting. See https://lightgbm.readthedocs.io/en/latest/Features.html

    Before training the model, we explore the parameter space for parameters with the least mean
    loss for the objective function (multi class log loss) when the parameters are used in cross
    validation.

    The parameter space is explored using the Tree of Parzen Estimators (TPE) algorithm. See
    http://hyperopt.github.io/hyperopt/#algorithms. 10 10-fold cross validation rounds are used to
    find the parameters resulting in the best mean log loss for a CV round. The cross validation
    used shuffles the data before splitting into test / train sets and is stratified, maintaining
    class probabilities in splits.

    Parameter tuning for LightGBM:
        https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        https://sites.google.com/view/lauraepp/parameters
    """
    label_encoder = LabelEncoder()
    train_set = _prepare_train_set(training_sets, features, label_encoder)
    print("###### Optimizing parameters ######")

    default_params = {"objective": "multiclassova",  # multiclass one vs all
                      "metrics": "softmax",  # multiclass log loss
                      "num_class": 4,
                      "nthread": cpu_count(),
                      "seed": seed,
                      "bagging_freq": 1,  # Perform bagging every kth iteration.
                      "early_stopping_rounds": 10,  # Maximum ensemble models without improvements.
                      "verbose": -1}

    def objective(hyper_params, n_folds=10):
        # Perform n_fold cross validation.
        all_params = {**hyper_params, **default_params}
        params = {**all_params}
        num_boost_round_ = params.pop("num_boost_round")
        early_stopping_rounds_ = params.pop("early_stopping_rounds")
        cv_results = lgb.cv(
            params,
            train_set,
            nfold=n_folds,
            verbose_eval=-1,
            num_boost_round=num_boost_round_,
            early_stopping_rounds=early_stopping_rounds_
        )
        best = cv_results["multi_logloss-mean"][-1]
        # Only show the parameters for this eval that are hyper params.
        print({k: all_params[k] for k in hyper_params.keys()})
        return {"loss": best, "params": all_params, "status": STATUS_OK}

    bayes_trials = Trials()
    space = {
        # Maximum leaves for each trained tree. Helps with overfitting when decreased.
        "num_leaves": hp.uniformint("num_leaves", 31, 41),
        # How much to change model in response to estimated error when the model weights are updated
        "learning_rate": hp.loguniform("learning_rate", np.log(0.05), np.log(0.1)),
        # Prune by minimum number of observations requirement. Helps with overfitting when increase.
        "min_child_samples": hp.uniformint("min_child_samples", 15, 23),
        # L1 regularization. Helps with overfitting when increased.
        "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        # L2 regularization. Helps with overfitting when increased.
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
        # Percentage of rows used per iteration frequency. Helps with overfitting when increase.
        "bagging_fraction": hp.uniform("bagging_fraction", 0, 1),
        # Max models in the ensemble. Increases accuracy when increased accuracy but may overfit.
        "num_boost_round": hp.uniformint("num_boost_round", 10, 100),
        # How much to weigh labels of positive class in OVA.
        "scale_pos_weight": hp.uniform("scale_pos_weight", 0, 10)
    }
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest,
                       max_evals=10, trials=bayes_trials)

    # Hyperopt returns floats even when using `uniformint`, so cast to int.
    best_params["num_leaves"] = int(best_params["num_leaves"])
    best_params["min_child_samples"] = int(best_params["min_child_samples"])
    best_params["num_boost_round"] = int(best_params["num_boost_round"])

    # Prepare training parameters.
    train_params = {**default_params, **best_params}
    num_boost_round = train_params.pop("num_boost_round")
    early_stopping_rounds = train_params.pop("early_stopping_rounds")

    print("###### Training Parameters ######")
    pprint(train_params)
    print("###### Training ######")

    # Train
    evals_result = {}  # To track training progress.
    fitted_model = lgb.train(
        train_params, train_set,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=-1,
        evals_result=evals_result, valid_sets=[train_set],  # Track training progress.
    )

    return fitted_model, label_encoder, evals_result
