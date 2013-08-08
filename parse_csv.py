import re
import logging
from collections import namedtuple


logging.getLogger().setLevel(logging.INFO)
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


def score(row):
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
    return sum(row[key] * coef for key, coef in coefs.iteritems())


def splitfields(line, sep=','):
    return [x.strip() for x in line.split(sep)]


def parse_file(stream):
    datarx = re.compile('^[0-9]')
    numrx = re.compile('^-?[0-9]+$')

    def is_numeric(s):
        return numrx.match(s)

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


def main():
    years = range(2008, 2013)
    # Parse all files
    year2data = {}
    for year in years:
        with open('fant%d.csv' % year, 'r') as stream:
            year2data[year] = parse_file(stream)

    assign_ids(year2data)


def assign_ids(year2data):
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
                elif ((name, team, year) in SPECIAL_CASE_TRADES):
                    source_nty = SPECIAL_CASE_TRADES[name, team, year]
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
    return


if __name__ == '__main__':
    main()
