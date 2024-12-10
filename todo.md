# code
## analysis tweaks
- reorder plots on `README.md`
- change boxplot y-axis to \[0,1\]
- make cutedges/percentage vs time plots to emphasize mixing
- run it for longer - even if I can demonstrate mixing, is 10k steps enough?

## visualization tweaks
- standardize x-axis ticks for 1:1 comparison across random/enacted
- side-by-side the random/enacted plots and cutedges/percentage vs time plots
- make a flow chart that demonstrates data processing pipeline

## future work - don't really have time for any of these, I don't think
- bring in census data and do demographic analysis via ensemble (hispanic)
- operationalizing Hispanic majority, 2% of ideal population level, not splitting cities, and not changing from previous map for optimization via acceptance/rejection chain
- environmental analysis of demographic data

# writing
## background
- partisan makeup of Clark County
    - note something about how land doesn't vote
- demographic majority-minority makeup of Clark County
- why the commission wanted to optimize for 2nd Hispanic district
    - how it relates to wider redistricting ideas
- partisan makeup of the Commission at time of redistricting process
- flesh out controversies around the redistricting process
    - mention legal stuff regarding 2021 redistricting in Nevada more broadly
    - demonstrate understanding of the legal landscape

## data and method
### data
- source and timeliness of data
- explain what a voter file is
- use of statewide election
    - population variance weirdness
    - turnout vs total population
    - whether commissioners were on the ballot
- trustworthiness of pipeline
### method
- mention `networkx` and `gerrychain` specifically
- explain the graph and why it matters that it remains connected
- summarize math behind my ensemble approach
    - random walks
    - random sampling
    - Markov chains
- explain why you chose this over different approaches to quantifying gerrymandering
- computational and data challenges I experienced
    - how they reflect wider problems related to redistricting

## discussion
- compare to outcomes for 2024 commission races
- compare to The Nevada Independent's analysis
- lack of demographic analysis
    - demonstrate knowledge by laying out how you would do the demographic analysis
- evaluation of my ensemble approach to quantifying gerrymandering
    - what does my analysis do well
    - what does it not do well
    - how can it be improved
    - can i make conclusions/recommendations based on my model

## conclusion
- make an argument about partisan gerrymandering and optimizing for Hispanic majority districts
- appendix that includes outputs of `data_cleaning.py` and `main.py`
