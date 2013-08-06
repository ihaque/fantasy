import sys
import re
import logging

logging.getLogger().setLevel(logging.INFO)
debug = logging.debug
info = logging.info
warning = logging.warning

# The logic should handle most trades properly, but in cases where there are two
# players with the same name in a previous year, it's hard to tell which one got
# traded; especially if only one of them plays in the latter year.
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
                    schema = [''.join(pair).strip().replace(' ', '_').replace('/', 'p')
                              if pair != ('','') else 'Name'
                              for pair in
                              zip(*map(splitfields, schemabuf))]
        else:
            assert schema is not None
            fields = line.replace('*','').replace('+','').split(',')
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

    # Try to assign a unique identifier to each player
    maxid = 0
    name2yearteamid = {}
    for year in years:
        for row in year2data[year]:
            name = row['Name']
            team = row['Tm']
            if name not in name2yearteamid:
                name2yearteamid[name] = [(year, team, maxid)]
                row['id'] = maxid
                maxid += 1
                debug('Created new entry for %s (%s %d)' % (name, team, year))
            else:
                yearteamids = name2yearteamid[name]
                # Is there a player with the same name and team in a previous year?
                # If so, update the last-seen year to this year and set the id.
                if (sum(1 for yti in yearteamids if
                        yti[0] < year and
                        yti[1] == team) == 1):
                    yti = next(yti for yti in yearteamids if
                               yti[0] < year and
                               yti[1] == team)
                    del yearteamids[yearteamids.index(yti)]
                    yearteamids.append((year, team, yti[2]))
                    row['id'] = yti[2]
                    info('Updating %s (%s, %d) to %d: %s' %
                         (name, team, yti[0], year, name2yearteamid[name]))
                # Is there exactly one other player with the same name this year?
                # Adrian Peterson and Steve Smith have players with the same name.
                elif (len(yearteamids) == 1 and
                    yearteamids[0][0] == year and
                    yearteamids[0][1] != team):
                    # Duplicate-name player
                    info('Creating new player for %s (%s, %d). Already saw '
                         '%s on %s in %d.' %
                         (name, team, year, name, yearteamids[0][1], year))
                    name2yearteamid[name].append((year, team, maxid))
                    row['id'] = maxid
                    maxid += 1
                # Was there exactly one player with this name in a previous year, on
                # a different team? Probably got traded.
                elif (sum(1 for yti in yearteamids if
                          yti[0] < year and
                          yti[1] != team) == 1):
                    yti = next(yti for yti in yearteamids if
                               yti[0] < year and
                               yti[1] != team)
                    info('Probable trade of %s from %s to %s between %d and %d' %
                         (name, yti[1], team, yti[0], year))
                    # Update last-seen buffer
                    del yearteamids[yearteamids.index(yti)]
                    yearteamids.append((year, team, yti[2]))
                    row['id'] = yti[2]
                elif ((name, team, year) in SPECIAL_CASE_TRADES):
                    source_nty = SPECIAL_CASE_TRADES[name, team, year]
                    # name is implicitly equal
                    yti = next(yti for yti in yearteamids if
                               yti[1] == source_nty[1] and
                               yti[0] == source_nty[2])
                    info('Special-case trade of %s from %s to %s between %d and %d' %
                         (name, yti[1], team, yti[0], year))
                    # Update last-seen buffer
                    del yearteamids[yearteamids.index(yti)]
                    yearteamids.append((year, team, yti[2]))
                    row['id'] = yti[2]
                else:
                    warning('could not assign %d %s %s %s' %
                            (year, name, team, yearteamids))
    return
    # Flatten by name/team
    for year1 in (2008, 2009, 2010, 2011):
        data1 = year2data[year1]
        data2 = year2data[year1 + 1]
        def key(row):
            return row['Name'], row['Tm']
        names1 = set(map(key, data1))
        names2 = set(map(key, data2))
        print year1
        print names1 ^ names2
if __name__ == '__main__':
    main()
