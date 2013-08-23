import logging
from sklearn import linear_model
import numpy
from operator import itemgetter

from parser import load_files
from prediction import construct_feature_matrix
from prediction import predict_scores
from prediction import ID, DELTA

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


def select_rows(collection, rows):
    return [collection[row] for row in rows]


def position_ranking_lists(identifiers, scores, id2name):
    positions = {position for name, team, position in id2name.itervalues()}
    deltas = {ident[DELTA] for ident in identifiers}
    delta2pos2list = {}
    for delta in deltas:
        pos2list = {}
        for position in positions:
            pos_idxs = [idx for idx, ident in enumerate(identifiers)
                        if (id2name[ident[ID]][2] == position and
                            ident[DELTA] == delta)]

            names = [id2name[ident[ID]][:2] for ident
                     in select_rows(identifiers, pos_idxs)]
            pos_scores = select_rows(scores, pos_idxs)
            pos2list[position] = sorted(zip(pos_scores, names),
                                        reverse=True)
        delta2pos2list[delta] = pos2list
    return delta2pos2list


def pos_rank_row_to_str(row):
    score = '% 6s' % ('%.2f' % row[0])
    name = '% 25s (% 3s)' % row[1]
    return '%s %s' % (score, name)


def compare_predictions(delta2pos2scores, delta2pos2preds, topN=100,
                        base_year=2013):
    for delta in delta2pos2scores:
        pos2scores = delta2pos2scores[delta]
        pos2preds = delta2pos2preds[delta]
        year = base_year - delta
        for position in sorted(pos2scores):
            print
            print '=============== %s (%d) ==============' % (position, year)
            print '% 32s\t% 32s' % ('Predicted', 'True')
            for i, (true, pred) in (
                enumerate(zip(pos2preds[position],
                              pos2scores[position])[:topN])):
                print '% 3d' % (i + 1),
                print '%s\t%s' % tuple(map(pos_rank_row_to_str, [true, pred]))
            print


def main():
    id2year2stats = load_files(
        {year: 'fant%d.csv' % year for year in xrange(2008, 2013)},
        SPECIAL_CASE_TRADES)

    def id_to_useful_name(id):
        year2stats = id2year2stats[id]
        any_year = year2stats[year2stats.keys()[0]]
        return (any_year['Name'], any_year['Tm'],
                any_year['FantasyFantPos'])

    matrix, identifiers, features = construct_feature_matrix(id2year2stats)
    id2name = {ident[ID]: id_to_useful_name(ident[ID]) for ident in
               identifiers}

    model = linear_model.LinearRegression()
    past_scores, past_predictions, current_predictions, current_ids = \
        predict_scores(matrix, identifiers, features, model)

    past_ranks = position_ranking_lists(identifiers, past_scores, id2name)
    past_predicted_ranks = position_ranking_lists(
        identifiers, past_predictions, id2name)
    current_predicted_ranks = position_ranking_lists(
        current_ids, current_predictions, id2name)

    compare_predictions(past_ranks, past_predicted_ranks)

    return

if __name__ == '__main__':
    main()
