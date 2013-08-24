from collections import defaultdict
from copy import copy
from itertools import chain
from logging import info

from numpy import array
from numpy import empty
from numpy import mean
from numpy import nan
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from constants import DELTA
from evaluation import compute_taus
from evaluation import position_ranking_lists


def score(row):
    """(Approximate) scoring function for my league"""
    coefs = {
        'PassingYds': 1.0/50,
        'PassingTD': 6,
        'PassingInt': -2,
        'RushingYds': 1.0/10,
        'RushingTD': 6,
        'ReceivingRec': 0.25,
        'ReceivingYds': 1.0/10,
        'ReceivingTD': 6,
        # Missing return TD, 2PC, Fumbles, FumRet
    }
    try:
        return sum((row[key] * coef if row[key] else 0)
                   for key, coef in coefs.iteritems())
    except:
        print [(key, row[key]) for key in coefs]
        raise


def isPosition(position):
    def predicate(row):
        return int(row['FantasyFantPos'] == position)
    return predicate


def stat_factory(key):
    def accessor(row):
        return nan if row[key] == "" else row[key]
    return accessor


# Stats that are just taken directly from last year's stats
# Use one-hot encoding for position
FIXED_STATS = [
    ('age', stat_factory('Age')),
    ('isQB', isPosition('QB')),
    ('isRB', isPosition('RB')),
    ('isWR', isPosition('WR')),
    ('isTE', isPosition('TE'))
]

# Stats that should be replicated by year
TRACKED_STATS = [
    ('games_played', stat_factory('G')),
    ('games_started', stat_factory('GS')),
    ('completions', stat_factory('PassingCmp')),
    ('pass_attempts', stat_factory('PassingAtt')),
    ('pass_yards', stat_factory('PassingYds')),
    ('pass_tds', stat_factory('PassingTD')),
    ('interceptions', stat_factory('PassingInt')),
    ('rush_attempts', stat_factory('RushingAtt')),
    ('rush_yards', stat_factory('RushingYds')),
    ('rush_tds', stat_factory('RushingTD')),
    ('rec_receptions', stat_factory('ReceivingRec')),
    ('rec_yards', stat_factory('ReceivingYds')),
    ('rec_tds', stat_factory('ReceivingTD')),
    ('fantasy_points', score),
]


def featurize_player(year2stats, id=None):
    """Construct a feature dictionary from the year-on-year stats for a player.

    Because different players have played for different lengths of time in
    the league (everyone from rookies, who will have had only one year of
    experience, to veterans, who will have as much experience as we have years
    in the database), it's not useful to just tag features with a particular
    year. It's probably most useful to tag stat features with a delta time
    from the current year (prev, prev-1, etc.) so that younger players just
    end up with a sparse feature vector.

    Some features (age, position) should just be taken from the last year
    and don't need to be replicated across years.
    """

    years = sorted(year2stats, reverse=True)
    last_year_stats = year2stats[years[0]]
    features = {}
    for feat_key, fn in FIXED_STATS:
        features[feat_key] = fn(last_year_stats)
    for year_idx, year in enumerate(years):
        year_delta = year_idx + 1
        stats = year2stats[year]
        for feat_key, fn in TRACKED_STATS:
            features[(feat_key, year_delta)] = fn(stats)

    if id is not None:
        features['id'] = id

    return features


def split_player(features):
    """

    Given a player's feature dictionary as computed by featurize_player,
    yield multiple feature dictionaries, one per year played, so that you
    predict each year independently in training.
    """
    def is_feature(stat_name):
        return (
            stat_name in [x[0] for x in FIXED_STATS] or
            any(stat_name[0] == x[0] for x in TRACKED_STATS))

    def is_identifier(stat_name):
        return not is_feature(stat_name)

    tracked = [feat for feat in features if type(feat) == tuple]
    tracked_features, deltas = zip(*tracked)
    deltas = set(deltas)
    fixed = [x[0] for x in FIXED_STATS]
    identifiers = [feat for feat in features if is_identifier(feat)]

    base_row = {(ident, 'identifier'): features[ident]
                for ident in identifiers}
    base_row.update({(fix, None): features[fix] for fix in fixed})
    for delta in deltas:
        # map the row at xx_d to name xx_(d-delta) in the new vector
        new_deltas = [(d, d - delta) for d in deltas if d >= delta]
        if len(new_deltas) == 1:
            continue
        new_row = copy(base_row)
        new_row[DELTA] = delta

        # At delta=1 we have the current age. Correct for the past.
        new_row[('age', None)] -= (delta - 1)

        for old_delta, new_delta in new_deltas:
            for feature in tracked_features:
                key = (feature, new_delta)
                new_row[key] = features[(feature, old_delta)]
        yield new_row


def test_split_player():
    features = {feat: idx for idx, (feat, fn) in enumerate(FIXED_STATS)}
    deltas = [1, 2, 3]
    features.update({('pass_tds', delta): delta for delta in deltas})

    split = list(split_player(features))
    split.sort(key=lambda row: row['delta'])
    expected = [
        {'age': 0, 'isQB': 1, 'isRB': 2, 'isWR': 3, 'isTE': 4,
         'delta': 1,
         'pass_tds_0': 1, 'pass_tds_1': 2, 'pass_tds_2': 3},

        {'age': -1, 'isQB': 1, 'isRB': 2, 'isWR': 3, 'isTE': 4,
         'delta': 2,
         'pass_tds_0': 2, 'pass_tds_1': 3},
    ]
    assert split == expected


def construct_feature_matrix(id2year2stats):
    feature_dicts = [featurize_player(year2stats, id) for id, year2stats in
                     id2year2stats.iteritems()]
    split_dicts = list(chain.from_iterable(split_player(features) for features
                                           in feature_dicts))

    def is_feature(feature_name):
        return feature_name[1] != 'identifier'

    keys = set(chain.from_iterable(split_dicts))
    feature_names = set(filter(is_feature, keys))
    identifier_names = keys - feature_names

    info('features:' + str(sorted(feature_names)))
    info('identifiers:' + str(sorted(identifier_names)))

    identifier_names = sorted(identifier_names)
    col2feature = sorted(feature_names)
    feature2col = {feature: idx for idx, feature in
                   enumerate(col2feature)}

    # By default, everything is a missing data point. Impute them away later.
    matrix = empty((len(split_dicts), len(feature_names)))
    matrix[:, :] = nan
    identifiers = []

    for row, instance in enumerate(split_dicts):
        identifiers.append(
            {ident: instance[ident] for ident in identifier_names})
        for feature in set(instance) & feature_names:
            matrix[row, feature2col[feature]] = instance[feature]

    return matrix, identifiers, col2feature


def cross_validate(matrix, identifiers, features, id2name, model, n_folds=3,
                   seed=None):
    """Use data from all year deltas > target_delta to predict scores."""
    feature_cols = [idx for idx, (feat, delta) in enumerate(features)
                    if delta != 0]
    objective_index = features.index(('fantasy_points', 0))

    def get_features_objective(_matrix):
        X = _matrix[:, feature_cols]
        y = _matrix[:, objective_index]
        return X, y

    taus_by_deltapos = {}
    accum_test_identifiers = []
    accum_test_scores = []
    accum_test_preds = []
    for fold, (train_index, test_index) in \
            enumerate(KFold(n=matrix.shape[0], n_folds=n_folds, shuffle=True,
                            random_state=seed)):
        imputer = Imputer()
        scaler = StandardScaler()
        train_matrix = matrix[train_index, :]
        test_matrix = matrix[test_index, :]
        imputer.fit(train_matrix)
        train_imputed = imputer.transform(train_matrix)
        train_imputed = scaler.fit_transform(train_imputed)
        test_imputed = scaler.transform(imputer.transform(test_matrix))

        X_train, y_train = get_features_objective(train_imputed)
        model.fit(X_train, y_train)
        X_test, y_test = get_features_objective(test_imputed)
        y_pred = model.predict(X_test)

        test_identifiers = [identifiers[idx] for idx in test_index]
        accum_test_identifiers.extend(test_identifiers)
        accum_test_scores.extend(y_test)
        accum_test_preds.extend(y_pred)

    pos_ranks_true = position_ranking_lists(
        accum_test_identifiers, accum_test_scores, id2name)
    pos_ranks_pred = position_ranking_lists(
        accum_test_identifiers, accum_test_preds, id2name)
    taus = compute_taus(pos_ranks_true, pos_ranks_pred)
    for deltapos, tauvals in taus.iteritems():
        taus_by_deltapos.setdefault(deltapos, []).append(tauvals)
    for deltapos in sorted(taus_by_deltapos, key=lambda x: (x[1], x[0])):
        print deltapos, taus_by_deltapos[deltapos]

    return


predict_scores = None

def predict_current_year(matrix, identifiers, features, id2name, model):
    imputed_matrix = Imputer().fit_transform(matrix)

    # Now, take the delta=1 rows (containing all our data) and make delta=0
    # rows by incrementing the delta indices for tracked stats and incrementing
    # age. This will be used with the trained model to predict this year.
    delta_1_indices = [idx for idx, ident in enumerate(identifiers)
                       if ident[DELTA] == 1]
    delta_1_rows = imputed_matrix[delta_1_indices, :]
    delta_1_dicts = (dict(zip(features, row)) for row in delta_1_rows)

    def shift_delta(feature_dict):
        # This is necessary because we do not set up any xxx_0 features
        # But, we select them when converting to rows, and then eliminate
        # them when subselecting columns. It's a kludge; could be avoided
        # by getting rid of the column subselect in this phase.
        rv = defaultdict(lambda: nan)
        for (feature, delta), value in feature_dict.iteritems():
            if delta is None:
                if feature == 'Age':
                    rv[feature, delta] = value + 1
                else:
                    rv[feature, delta] = value
            else:
                rv[feature, delta + 1] = value
        return rv

    delta_0_dicts = (shift_delta(row) for row in delta_1_dicts)
    delta_0_rows = [[row[feature] for feature in features] for row in
                    delta_0_dicts]
    delta_0_matrix = array(delta_0_rows)
    current_year_predictions = model.predict(delta_0_matrix[:, feature_cols])
    current_year_idents = []
    for idx in delta_1_indices:
        current_year_idents.append(copy(identifiers[idx]))
        current_year_idents[-1][DELTA] = 0

    return Y, y_pred, current_year_predictions, current_year_idents
