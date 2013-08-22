import logging
from sklearn import linear_model
import numpy
from operator import itemgetter

from parser import load_files
from prediction import construct_feature_matrix

logging.getLogger().setLevel(logging.ERROR)
debug = logging.debug
info = logging.info
warning = logging.warning
error = logging.error

# The logic should handle most trades properly, but in cases where there are
# two players with the same name in a previous year, it's hard to tell which
# one got traded; especially if only one of them plays in the latter year.
SPECIAL_CASE_TRADES = {
    ('Zach Miller', 'SEA', 2011): ('Zach Miller', 'OAK', 2010)
}


def select_columns(matrix, keyidxs):
    columns = [matrix[:, idx].reshape(matrix.shape[0], 1) for idx in keyidxs]
    return numpy.hstack(columns)


def predict(id2year2stats, feature_matrix, key2idx, features, model):
    indices = [key2idx[key] for key in features]
    X = select_columns(feature_matrix, indices)
    y = select_columns(feature_matrix, [key2idx['fantasy_points_1']])

    model.fit(X, y)
    y_pred = model.predict(X)
    id_idx = key2idx['id']
    predictions = []
    for row in xrange(y_pred.shape[0]):
        id = feature_matrix[row, id_idx]
        year2stats = id2year2stats[id]
        years = sorted(year2stats)
        name = year2stats[years[-1]]['Name']
        team = year2stats[years[-1]]['Tm']
        position = year2stats[years[-1]]['FantasyFantPos']
        predictions.append((name, position, team, y_pred[row], y[row]))

    return predictions


def position_ranking_lists(predictions):
    positions = sorted({pos for name, pos, team, yp, y in predictions})
    for position in positions:
        rows = [pred for pred in predictions if pred[1] == position]
        by_pred = sorted(rows, key=itemgetter(3), reverse=True)
        by_actual = sorted(rows, key=itemgetter(4), reverse=True)
        print
        print "==== %s ====" % position
        for idx, (bp, ba) in enumerate(zip(by_pred[:50], by_actual[:50])):
            print "%03d % 25s %.2f   %.2f % 25s" % (
                idx + 1, bp[0], bp[3], ba[4], ba[0])


def main():
    id2year2stats = load_files(
        {year: 'fant%d.csv' % year for year in xrange(2008, 2013)},
        SPECIAL_CASE_TRADES)

    def id2name(id):
        year2stats = id2year2stats[id]
        any_year = year2stats.keys()[0]
        return (year2stats[any_year]['Name'], year2stats[any_year]['Tm'])

    matrix, identifiers, features = construct_feature_matrix(id2year2stats)
    return

    features = [key for key in key2idx if
                key != 'id' and not key.endswith('_1')]

    predictions = predict(id2year2stats, feature_matrix, key2idx, features,
                          linear_model.Lasso(max_iter=100000))
    position_ranking_lists(predictions)

if __name__ == '__main__':
    main()
