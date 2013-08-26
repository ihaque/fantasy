# Fantasy Football Draft Predictor

** The results from this code are not very good. Treat it as an
   interesting experiment, not as a legitimate source of fantasy
   rankings.**

This code uses machine learning techniques to attempt to predict fantasy
football performance of players based on their performance in past years.
Scoring is customizable by league settings, so that predicted points and
rankings are tailored to a league's particular desires.

## Data

### Source and Quality

All data was gathered from pro-football-reference.com CSV dumps. This
data has the great advantage of being easily available and CSV-formatted,
but is not great. In particular, it's missing important stat categories like
fumbles and fumble recoveries (and less significantly, 2-point conversions).
It's also useless for leagues with point bonuses for big games or big plays
(eg, bonuses at 100/200/etc. yards, or 40+ yard plays), because the stats are
aggregated over an entire season.

### Parsing

Parsing routines for this data are implemented in parsing.py.

Merely getting data out of the files is pretty easy; with the exception of
headers that recur and are split over two lines (which requires the minimally
stateful parser in `parser._parse_file`), and some extra sigils around players
with interesting properties, the data are easy to get. There are instances of
missing data, represented as empty strings.

The interesting work in the parser is that required to construct a unified data
set from multiple years' worth of files. In particular, the CSV dumps do NOT
contain a unique ID for each distinct player. To do meaningful learning, we must
be able to track which player is which from year to year, even in the face of
team changes, new players coming in the with same name, and in principle name
changes as well. (Thankfully, the data set I used did not make me unify Chad
Johnson and Chad Ochocinco.)

This work is all done in `parser._assign_ids`. The ID assignment goes through
the parsed data in chronological order. If a player with a particular name has
never been seen before, it assumes this is a new unique player, and assigns a
new ID to this player (tagging it with position, team, and last year seen
playing). Things get more interesting once we start seeing the same names show
up.

To explain the logic, let's say we see a player named N, playing position P,
on team T, in year Y. If we saw a player N playing P on T in year Y-1,
we assume this is the same player. This is the common case of a player
staying on one team from year to year.

Occasionally, you'll see two players both named N, but playing for different
teams in the same year Y (for example, consider the Adrian Petersons of the
Vikings and of the Bears, or the Steve Smiths of the Giants and the Panthers).
These will be considered separate players.

If there was exactly one player sharing N and P in a previous year, but on a
different team the previous year, then that player was probably traded, so we
assign him the ID from that previous player record. Once in a blue moon, you'll
even see a player who keeps the same team, but appears to change position
(in the database, this happens to Steve Slaton changing from RB in 2008 to
WR in 2009); this will also be marked as the same player.

Those two clauses have a nasty interaction, which is that you may have two
players with the same name who "switch" both team and position: for example,
consider Alex Smith (TE, TB) and Alex Smith (QB, SF). These cases are
specially marked to split the two players.

Finally, because the ID assignment assigns IDs sequentially as it goes through
the parsed file, rather than simultaneously, there are some trades which may not
be possible to disambiguate except manually. The one case in which this appeared
was for Zach Miller (TE), traded from OAK to SEA between 2010 and 2011. There
was also a Zach Miller (TE) playing for JAC in 2010. If the entire file were
parsed, it would be seen that the latter Miller stayed at Jacksonville, so the
likelier parse would be that the Oakland Miller moved to Seattle. In principle,
though, nothing rules out a JAC->SEA and OAK->JAC trade sequence, so this case
is explicitly coded (via a `SPECIAL_CASE_TRADES` dict in constants.py).


## Learning and Evaluation

Code in prediction.py and evaluation.py.

### Featurization

Each player is described by a single feature vector (but notice that this later
gets rotated and split in the learning). All features are numeric, and there are
two classes of features:

 - "Fixed" features only get one copy in the feature vector, containing the
   value corresponding to the most recent year of data. These are currently
   age (in years) and position (four dimensions corresponding to one-hot
   encoding of QB, WR, RB, or TE).

 - "Tracked" features recur once per year that we have data (eg, if data
   corresponding to five years is passed into the file, then there will be
   five copies of each tracked feature, corresponding to each year loaded.
   For players that don't show up in every year (particularly younger players
   or players who retire), there will be missing data, represented as
   absence from the dictionary.

Both types of features are computed by arbitrary Python functions of the data
parsed out of the data file; the tracked features used consist of the stats
used to compute fantasy scores, as well as the fantasy score itself.

Assume that we're
in 2013, and so have data files for 2012 and 2011. Each tracked feature is
replicated twice, with a tag `year_delta` corresponding to how many years back
the data is from the present. For example, "Yards passing, delta=1" would be
passing yards in 2012. This means that a player who last played in 2011
would only have entries in "Yards passing, delta=2"; a 2012 rookie would only
have delta=1, and a veteran might have both.

`prediction.featurize_player` takes a stats dictionary for a player (as built
by the parser) and builds a feature dictionary for that player.

### Instance Splitting

The objective of the model is to predict the number of fantasy points scored
by each player this year, by leveraging all past data we have on that player.
To train and validate the model, we would like to predict the number of fantasy
points scored in previous years. However, it is not sensible to use, say,
2012 data to try to predict 2011.

`prediction.split_player` takes a feature dictionary for a player and emits  
a new feature dictionary for each year that that player played. To understand
its logic, it's easiest to consider an example. Let's say we have a 30-year-old
QB who played in 2011 and 2010, and we have data for 2012, 2011, and 2010.
Let's only track age, position, and passing yards (PassYd for short). This
player might have the following features coming from the initial featurization
pass:

 - Age: 30
 - isQB: 1; all other positions 0
 - PassYd, 1: does not exist (did not play in 2012)
 - PassYd, 2: 1200 [2011 yards]
 - PassYd: 3: 1500 [2011 yards]

The splitting code copies all fixed features to each new row (with a
special-case correction for age). A new row would then be emitted for each
year:
 - Age 30; QB; (PassYd, 0): 1200; (PassYd, 1): 1500, (PassYd 2-3): missing
 - Age 29; QB; (PassYd, 0): 1500; (PassYd, 1-3): missing

This data can then be used to build a uniform predictor that jointly trains over
all years: we can just build a model to predict on the target property at
year delta 0. The given example is obviously trivial, since only one instance
actually has much data beyond the objective (PassYd, 0), but the real system
would shift and replicate all the tracked stats.

Once the splitting is complete, the split feature dictionaries are easily turned
into a feature matrix in `prediction.construct_feature_matrix`. Columns are
created for the union of all features present in the given instances; missing
entries are encoded as NaN to be resolved later in the pipeline.

### Learning and Validation

Given the matrix form of features, learning is a straightforward regression
problem. The learning pipeline, implemented separately in both
`prediction.cross_validate` and `prediction.predict_current_year` (DRY
violation, but I was running up against the draft deadline!), first fills in
missing features by mean-value imputation. It then optionally does zero-mean,
unit-variance standardization. This step is necessary for some learning
algorithms (e.g., support vector regression), but complicates interpretation of
the output data unless you undo the scaling. It turns out SVR doesn't actually
work that well, it's annoying to back out the shift and scale for fantasy
points, and the other algorithms mostly don't care, so the main code
does not use standardization.

`prediction.cross_validate` uses k-fold cross validation to test the performance
of the model(s). The input data is split into k-folds; on each fold we fit the
imputer, standardizer, and model only on the training data and apply them to
both training and test data. The regressed scores for each fold are accumulated
until at the end of the cross-validation, we have test predictions for every
player. The unified lists of true and predicted scores are then split into lists
by player position by `prediction.position_ranking_lists`.

As a first objective, we're more interested in the relative
positioning of players than the absolute points difference. A learning-to-rank
method might be better at this problem, but this is an easy substitute and
eventually seeing predicted fantasy points is useful as well.

I evaluate the quality of the model by computing Kendall's tau score between
the true and predicted scores for each position in each year (eg, 2008 QB true
and predicted ordering). [Kendall's tau coefficient](http://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient)
is a nonparametric statistical test for the dependence of two different
variables. If the rankings produced by the two agree exactly, tau=1; if they
disagree perfectly (one is the reverse of the other), tau=-1; if they are
independent, tau=0. Thus, we prefer models with cross-validated tau near 1.
If the tau is near 0, then the model is learning nothing. 
`prediction.compute_taus` evaluates the tau scores given the ranking lists for
each year. (NB: it's not actually each year -- it's the last year a given player
played. So, for example, players who played in 2012 may appear in the same bin
as retired players from earlier. This should be OK, since we'll have the true
fantasy points for the same set of players.)

Since there are a lot of players who are completely irrelevant for fantasy
purposes, only the ranking on the high-rank players is of interest. (We
would like the model to distinguish Drew Brees and Mark Sanchez, but we really
don't care about 4th-string quarterbacks.) Thus, tau is actually computed on
a subset of the data set: only the top N (configurable in `constants.TOP_N`)
players in each year at each position. We thus report two
parameters for each year/position/model triple: the fraction of players in
the true top N who ranked in the predicted top N, and the tau coefficient
for this intersection list.

## Performance

`main.main` is a driver script that loads the data, featurizes it, displays
cross-validation results from a number of models, and then dumps predicted
scores and rankings for the current year. I tried three major classes of
regression algorithms for this problem:

 - Generalized linear models
  + Run-of-the-mill linear regression
  + L2-regularized "ridge" regression
 - Ensemble methods
  + Random forest regression
  + Extremely randomized trees regression
  + Gradient boosted regression trees
  + AdaBoost.R2 regression
 - Support vector regression
  + C-regularized, RBF kernel
  + nu-regularized, RBF kernel

There was no particular basis for selecting these models. I don't particularly
believe that a pure linear model would work, but it's the easiest possible model
to try. Ensemble methods are not particularly interpretable, but do have good
practical and theoretical performance. SVMs are theoretically beautiful and a
decent shot at a good learning model. Mostly I was guided by what was
conveniently available in off-the-shelf format from scikit-learn! I made
no effort at hyperparameter optimization. While this is known to have an effect
on some models (eg, SVMs), the performance of all methods was sufficiently
similar that I didn't expect any magical gains from tuning.

Randomization in cross-validation and ensemble methods means that particular
results vary from run to run, but the overall sketch of performance remains
fairly similar. Let's take a look at Kendall's tau and the fraction of the top
50 found in the top 50 predicted results for each position, only in the most
recent year for each player (ie, the entries with the largest amount of data).
Note that the cross-validation fold splits should be the same between models,
and I used 10-fold CV.

<table>
<tr><th>Algorithm</th><th>QB</th><th>RB</th><th>TE</th><th>WR</th></tr>
<tr><td>Linear regression</td>
    <td>0.505, 76%</td>
    <td>0.379, 54%</td>
    <td>0.301, 70%</td>
    <td>0.273, 68%</td></tr>
<tr><td>Ridge regression</td>
    <td>0.549, 78%</td>
    <td>0.360, 56%</td>
    <td>0.308, 70%</td>
    <td>0.291, 68%</td></tr>
<tr><td>Random forests</td>
    <td>0.533, 72%</td>
    <td>0.191, 60%</td>
    <td>0.287, 70%</td>
    <td>0.144, 60%</td></tr>
<tr><td>Extremely randomized trees</td>
    <td>0.544, 74%</td>
    <td>0.266, 58%</td>
    <td>0.152, 68%</td>
    <td>0.166, 62%</td></tr>
<tr><td>AdaBoost.R2</td>
    <td>0.511, 74%</td>
    <td>0.455, 62%</td>
    <td>0.285, 56%</td>
    <td>0.255, 56%</td></tr>
<tr><td>GBRT</td>
    <td>0.531, 76%</td>
    <td>0.360, 56%</td>
    <td>0.374, 74%</td>
    <td>0.217, 62%</td></tr>
<tr><td>C-SVR</td>
    <td>0.319, 68%</td>
    <td>0.222, 56%</td>
    <td>0.251, 72%</td>
    <td>0.148, 66%</td></tr>
<tr><td>nu-SVR</td>
    <td>0.301, 68%</td>
    <td>0.276, 58%</td>
    <td>0.244, 72%</td>
    <td>0.163, 66%</td></tr>
</table>

There are a handful of takeaways from this table:

 - SVMs perform noticeably worse than any other method. This might be down
 to the lack of hyperparameter tuning (eg, C/nu fitting).
 - Regularization on linear regression doesn't make a huge difference. Ridge
 (L2 regularized) linear regression performs about the same as the
 unregularized standard algorithm. The feature vectors here are not very
 high-dimensional, though there is some linear dependence (eg, fantasy points
 are a linear combination of other terms).
 - Boosting methods performed noticeably better than the random-forest methods.
 AdaBoost.R2 and GBRTs both had far better performance ranking RBs and WRs
 (the positions with more active players than others) than did random forests
 or extremely randomized trees.

The main takeaway, though, is that despite small differences in performance
among methods, nothing actually works very well. The best tau scores we see are
on the order of 0.5, for QBs down to 0.3 for WRs. While we can sort-of predict
who the top 50 players at each position will be (getting maybe 35 out of 50
right), the rankings within those 50 are only a little better than random.
Crucially, they're probably not much better than just looking at ESPN or Yahoo
rankings (or just looking at some names, if you pay attention to football at
all).

## Conclusion

This was a fun exercise, but not a terribly useful project in terms of actual
performance. It's an interesting demonstration of applying non-time-series
methods to predict time-series data for a practical problem.

An interesting enhancement to the model would be to consider team membership as
well. A very simple version would be to just add team as a feature; this just
considers local "dynasties" where some teams are better than others, period. A
more refined version would consider who else a player is playing with. For
example, if a quarterback's receivers all get traded away, then it is reasonable
to guess that his passing performance will drop off. However, this requires
consideration of cross-player considerations, and is not easily integrated into
this model (running it as a second stage of classification might work).
Integrating injury data and quality of a team's defense may also be
interesting sources of information.
