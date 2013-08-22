import re
import logging
from collections import namedtuple
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn import linear_model
import numpy
from operator import itemgetter

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


def string_safe(st):
    return st if st else 0

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
        return sum((row[key] * coef if row[key] else 0) for key, coef in coefs.iteritems())
    except:
        print [(key, row[key]) for key in coefs]
        raise


def parse_file(stream):
    """Parse CSV fantasy data from http://www.pro-football-reference.com/"""
    datarx = re.compile('^[0-9]')
    numrx = re.compile('^-?[0-9]+$')

    def is_numeric(s):
        return numrx.match(s)

    def splitfields(line, sep=','):
        return [x.strip() for x in line.split(sep)]

    schema = None
    schemabuf = []
    rows = []
    for line in stream:
        line = line.strip()
        if not datarx.match(line):
            if schema:
                continue
            else:
                schemabuf.append(line)
                if len(schemabuf) == 2:
                    # reconstruct schema from two rows
                    schema = [(''.join(pair).strip()
                                 .replace(' ', '_').replace('/', 'p'))
                              if pair != ('', '') else 'Name'
                              for pair in
                              zip(*map(splitfields, schemabuf))]
        else:
            assert schema is not None
            fields = line.replace('*', '').replace('+', '').split(',')
            rows.append({key: (float(val) if is_numeric(val) else val)
                         for key, val in zip(schema, fields)})

    return rows


def assign_ids(year2data, special_case_trades={}):
    """Attempt to identify unique players and assign each a primary key.

    Looks at the list of player stats for each year and attempts to:
        1. Identify unique players in each year
        2. Track players across years
    in order to assign each player a primary key that follows them in time.

    Can handle multiple players in the same season with the same name, and can
    follow trades (if position stays constant) and position changes (if team
    stays constant). Some trades are ambiguous, and can be special-cased
    with the `special_case_trades` parameter:

        { (Name, NewTeam, NewYear): (Name, OldTeam, OldYear)}

    For example:
        ('Zach Miller', 'SEA', 2011): ('Zach Miller', 'OAK', 2010)

    Represents a trade of Zach Miller from OAK in 2010 to SEA in 2011. This
    is needed because there was also a Zach Miller in JAX in 2010, so without
    the special case, it's hard to tell which one got traded to SEA. (Both are
    TEs.) It could be done by looking at all the rows in this year to see
    if one of them stayed put (ZM on JAX did), but the parser here does not
    consider all of that context -- it may not have seen the JAX row when it
    encounters the SEA row.
    """
    _playerkey = namedtuple(
        '_playerkey',
        ('year', 'team', 'id', 'name', 'position'))

    # Try to assign a unique identifier to each player
    maxid = 0
    name2keys = {}
    for year in sorted(year2data):
        for row in year2data[year]:
            name = row['Name']
            team = row['Tm']
            position = row['FantasyFantPos']

            # Is there a player with the same name and team in a previous
            # year?
            def same_team(key):
                return (key.year < year and
                        key.team == team and
                        key.name == name and
                        key.position == position)

            # Is there exactly one other player with the same name this
            # year? eg, Adrian Peterson and Steve Smith each are names
            # that represent two different players.
            # Alternatively, a player with the same name in an earlier year,
            # but in a different position and team (Alex Smith TE vs QB).
            def doppelganger(key):
                return ((key.year == year and
                         key.team != team and
                         key.name == name) or
                        (key.year < year and
                         key.team != team and
                         key.name == name and
                         key.position != position))

            # Was there exactly one player with this name and the same
            # position in a previous year, on a different team?
            # Probably got traded.
            def traded(key):
                return (key.year < year and
                        key.team != team and
                        key.position == position)

            # Did the player change positions on the same team? e.g.
            # Steve Slaton 2008 RB -> 2009 WR
            def position_change(key):
                return (key.year < year and
                        key.team == team and
                        key.name == name and
                        key.position != position)

            if name not in name2keys:
                name2keys[name] = [_playerkey(year, team, maxid,
                                              name, position)]
                row['id'] = maxid
                maxid += 1
                debug('Created new entry for %s (%s %d)' % (name, team, year))
            else:
                keys = name2keys[name]

                def update_last_seen_and_row(oldkey):
                    row['id'] = oldkey.id
                    del keys[keys.index(oldkey)]
                    newkey = oldkey._replace(year=year, team=team)
                    keys.append(newkey)

                # Same player, new year.
                if (len(filter(same_team, keys)) == 1):
                    key = filter(same_team, keys)[0]
                    update_last_seen_and_row(key)
                    info('Updating %s (%s, %d) to %d: %s' %
                         (name, team, key.year, year, name2keys[name]))

                # Same name, different player.
                elif (len(filter(doppelganger, keys)) == 1
                      and len(keys) == 1):
                    # Duplicate-name player
                    info('Creating new player for %s (%s, %d). Already saw '
                         '%s on %s in %d.' %
                         (name, team, year, name, keys[0].team, year))
                    keys.append(_playerkey(year, team, maxid, name, position))
                    row['id'] = maxid
                    maxid += 1

                # Traded players.
                elif len(filter(traded, keys)) == 1:
                    key = filter(traded, keys)[0]
                    info('Probable trade of %s from %s to %s between %d/%d' %
                         (name, key.team, team, key.year, year))
                    update_last_seen_and_row(key)

                # Special-cased trades.
                elif ((name, team, year) in special_case_trades):
                    source_nty = special_case_trades[name, team, year]
                    # name is implicitly equal
                    key = next(key for key in keys if
                               key.team == source_nty[1] and
                               key.year == source_nty[2])
                    info('Special-case trade of %s from %s to %s bw %d/%d' %
                         (name, key.team, team, key.year, year))
                    update_last_seen_and_row(key)

                # Players who changed position
                elif len(filter(position_change, keys)) == 1:
                    key = filter(position_change, keys)[0]
                    info('%s probably changed position from %s to %s on'
                         '%s from %d to %d' %
                         (name, key.position, position, team, key.year, year))
                    update_last_seen_and_row(key)

                else:
                    error('could not assign %d %s %s %s %s' %
                          (year, name, team, position, keys))
            assert 'id' in row, str(row)
    return


def featurize(year2stats, id=None):
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
    # Stats that are just taken directly from last year's stats
    FIXED_STATS = [('age', 'Age'), ('position', 'FantasyFantPos')]

    # Stats that should be replicated by year
    TRACKED_STATS = [
        ('games_played', 'G'),
        ('games_started', 'GS'),
        ('completions', 'PassingCmp'),
        ('pass_attempts', 'PassingAtt'),
        ('pass_yards', 'PassingYds'),
        ('pass_tds', 'PassingTD'),
        ('interceptions', 'PassingInt'),
        ('rush_attempts', 'RushingAtt'),
        ('rush_yards', 'RushingYds'),
        ('rush_tds', 'RushingTD'),
        ('rec_receptions', 'ReceivingRec'),
        ('rec_yards', 'ReceivingYds'),
        ('rec_tds', 'ReceivingTD'),
    ]
    # Stats that should be replicated, but are functions of the entire stats
    TRACKED_COMPUTED_STATS = [
        ('fantasy_points', score),
    ]

    # Computed stats are strictly more powerful than directly copied stats,
    # but the latter are more convenient to specify, so I implement the
    # latter in terms of the former.
    def stat_factory(key):
        return lambda row: string_safe(row[key])

    TRACKED_COMPUTED_STATS.extend(
        (feature_key, stat_factory(row_key))
        for feature_key, row_key in TRACKED_STATS)

    years = sorted(year2stats, reverse=True)
    last_year_stats = year2stats[years[0]]
    features = {}
    for feat_key, row_key in FIXED_STATS:
        features[feat_key] = string_safe(last_year_stats[row_key])
    for year_idx, year in enumerate(years):
        year_delta = year_idx + 1
        stats = year2stats[year]
        for feat_key, fn in TRACKED_COMPUTED_STATS:
            features['%s_%d' % (feat_key, year_delta)] = fn(stats)

    if id is not None:
        features['id'] = id

    return features


def construct_feature_matrix(id2year2stats):
    rows = [featurize(year2stats, id) for id, year2stats in
            sorted(id2year2stats.iteritems())]
    keys = sorted({key for row in rows for key in row})
    positions = sorted({row['position'] for row in rows})
    key2idx = {key: idx for idx, key in enumerate(keys)}
    #X = dok_matrix((len(rows), len(key2idx)))
    X = numpy.zeros((len(rows), len(key2idx)))
    print X.shape
    for i, row in enumerate(rows):
        for key in row:
            if key == 'position':
                X[i, key2idx[key]] = positions.index(row[key])
            else:
                try:
                    X[i, key2idx[key]] = row[key]
                except:
                    print key
                    print row
                    raise
    return X, key2idx


def select_columns(matrix, keyidxs):
    submatrix = dok_matrix((matrix.shape[0], len(keyidxs)))
    columns = [matrix[:, idx].reshape(matrix.shape[0], 1) for idx in keyidxs]
    #return sparse_hstack(columns)
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
    years = range(2008, 2013)
    # Parse all files
    year2data = {}
    for year in years:
        with open('fant%d.csv' % year, 'r') as stream:
            year2data[year] = parse_file(stream)

    assign_ids(year2data, SPECIAL_CASE_TRADES)
    # Transpose to get years by player
    id2year2stats = {}
    for year, data in year2data.iteritems():
        for datum in data:
            year2stats = id2year2stats.setdefault(datum['id'], {})
            year2stats[year] = datum

    feature_matrix, key2idx = construct_feature_matrix(id2year2stats)
    features = [key for key in key2idx if
                key != 'id' and  not key.endswith('_1')]

    predictions = predict(id2year2stats, feature_matrix, key2idx, features,
                          linear_model.Lasso(max_iter=100000))
    position_ranking_lists(predictions)

if __name__ == '__main__':
    main()
