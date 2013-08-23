from scipy.stats import kendalltau

from constants import BASE_YEAR
from constants import DELTA
from constants import ID
from constants import TOP_N


def select_rows(collection, rows):
    return [collection[row] for row in rows]


def kendall_tau(position_scores, position_predictions, topN=TOP_N):
    """
    Each arg has form [(score, (name, team, id))].

    Extract IDs from each, find intersection, remap to unique IDs in [0,N), and
    use scipy.
    """

    def get_ids(score_list):
        return [id for score, (name, team, id) in score_list[:topN]]

    true_ids = get_ids(position_scores)
    pred_ids = get_ids(position_predictions)
    shared = set(true_ids) & set(pred_ids)
    frac_shared = float(len(shared)) / topN

    def get_scores(score_list):
        # Sort to ensure same order among lists
        idscore = sorted([(id, score) for score, (name, team, id)
                          in score_list if id in shared])
        return [score for id, score in idscore]
    true_scores = get_scores(position_scores)
    pred_scores = get_scores(position_predictions)

    return kendalltau(true_scores, pred_scores), frac_shared


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

            names = [id2name[ident[ID]][:2] + (ident[ID],) for ident
                     in select_rows(identifiers, pos_idxs)]
            pos_scores = select_rows(scores, pos_idxs)
            pos2list[position] = sorted(zip(pos_scores, names),
                                        reverse=True)
        delta2pos2list[delta] = pos2list
    return delta2pos2list


def pos_rank_row_to_str(row):
    score = '% 6s' % ('%.2f' % row[0])
    name = '% 25s (% 3s)' % row[1][:2]
    return '%s %s' % (score, name)


def compare_predictions(delta2pos2scores, delta2pos2preds, topN=TOP_N,
                        base_year=BASE_YEAR):
    for delta in delta2pos2scores:
        pos2scores = delta2pos2scores[delta]
        pos2preds = delta2pos2preds[delta]
        year = base_year - delta
        for position in sorted(pos2scores):
            (tau, pval), frac_shared = kendall_tau(pos2scores[position],
                                                   pos2preds[position])
            print
            print '=================== %s (%d) ==================' % \
                (position, year)
            print 'Kendall Tau: %f (p=%f), %.2f%% shared' % (tau, pval,
                                                             100 * frac_shared)
            print '% 32s\t% 32s' % ('Predicted', 'True')
            for i, (true, pred) in (
                enumerate(zip(pos2preds[position],
                              pos2scores[position])[:topN])):
                print '% 3d' % (i + 1),
                print '%s\t%s' % tuple(map(pos_rank_row_to_str, [true, pred]))
            print
