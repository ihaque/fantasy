import logging

from constants import BASE_YEAR
from constants import TOP_N
from constants import SPECIAL_CASE_TRADES
from evaluation import compare_predictions
from evaluation import pos_rank_row_to_str
from evaluation import position_ranking_lists
from parser import load_files
from prediction import construct_feature_matrix
from prediction import predict_scores
from prediction import ID


logging.getLogger().setLevel(logging.ERROR)
debug = logging.debug
info = logging.info
warning = logging.warning
error = logging.error


def dump_predictions(delta2pos2preds, topN=TOP_N, base_year=BASE_YEAR):
    for delta in delta2pos2preds:
        pos2preds = delta2pos2preds[delta]
        year = base_year - delta
        for position in sorted(pos2preds):
            print
            print '=============== %s (%d) ==============' % (position, year)
            print 'Predicted'
            for i, pred in enumerate(pos2preds[position][:topN]):
                print '% 3d' % (i + 1),
                print '%s' % pos_rank_row_to_str(pred)
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

    #from sklearn import linear_model
    from sklearn import ensemble
    #from sklearn import svm
    #model = linear_model.LinearRegression()
    #model = linear_model.Lasso(max_iter=100000)
    #model = ensemble.RandomForestRegressor()
    model = ensemble.GradientBoostingRegressor()

    past_scores, past_predictions, current_predictions, current_ids = \
        predict_scores(matrix, identifiers, features, model)

    past_ranks = position_ranking_lists(identifiers, past_scores, id2name)
    past_predicted_ranks = position_ranking_lists(
        identifiers, past_predictions, id2name)
    current_predicted_ranks = position_ranking_lists(
        current_ids, current_predictions, id2name)

    compare_predictions(past_ranks, past_predicted_ranks)
    dump_predictions(current_predicted_ranks)

    return

if __name__ == '__main__':
    main()
