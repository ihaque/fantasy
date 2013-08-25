import logging
from random import randint

from constants import BASE_YEAR
from constants import ID
from constants import TOP_N
from constants import SPECIAL_CASE_TRADES
from evaluation import pos_rank_row_to_str
from evaluation import position_ranking_lists
from parser import load_files
from prediction import construct_feature_matrix
from prediction import cross_validate
from prediction import predict_current_year


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

    current_players = set(id for id in id2year2stats if BASE_YEAR - 1 in
                          id2year2stats[id])

    matrix, identifiers, features = construct_feature_matrix(id2year2stats)
    id2name = {ident[ID]: id_to_useful_name(ident[ID]) for ident in
               identifiers}

    from sklearn import linear_model
    from sklearn import ensemble
    from sklearn import svm

    seed = randint(0, 2**32 - 1)
    for model in [linear_model.LinearRegression(),
                  linear_model.Ridge(),
                  ensemble.RandomForestRegressor(),
                  ensemble.ExtraTreesRegressor(),
                  ensemble.AdaBoostRegressor(),
                  ensemble.GradientBoostingRegressor(),
                  svm.SVR(),
                  svm.NuSVR(),
                  ]:
        print str(model).split('(')[0]
        cross_validate(matrix, identifiers, features, id2name, model,
                       n_folds=10, seed=seed)
        print

    model = ensemble.RandomForestRegressor()
    current_predictions, current_ids = \
        predict_current_year(matrix, identifiers, features, id2name, model)

    current_predictions, current_ids = zip(
        *[(pred, ident) for pred, ident
          in zip(current_predictions, current_ids)
          if ident[ID] in current_players])

    current_predicted_ranks = position_ranking_lists(
        current_ids, current_predictions, id2name)

    dump_predictions(current_predicted_ranks)

    return

if __name__ == '__main__':
    main()
