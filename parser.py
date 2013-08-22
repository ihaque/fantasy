import re
from collections import namedtuple
from logging import debug
from logging import error
from logging import info


def _parse_file(filename):
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
    with open(filename, 'r') as stream:
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


def _assign_ids(year2data, special_case_trades):
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


def load_files(year2filename, special_case_trades={}):
    year2data = {year: _parse_file(fn) for year, fn in
                 year2filename.iteritems()}
    _assign_ids(year2data, special_case_trades)

    # Transpose to get years by player
    id2year2stats = {}
    for year, data in year2data.iteritems():
        for datum in data:
            year2stats = id2year2stats.setdefault(datum['id'], {})
            year2stats[year] = datum

    return id2year2stats
