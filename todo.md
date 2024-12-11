# code
## analysis tweaks
YES - reorder plots on `README.md`
YES - update `.gitignore`
YES - upload data as `.zip`
YES - change boxplot y-axis to \[0,1\]
YES - make cutedges/percentage vs time plots to emphasize mixing - something to emphasize mixing
        - these basically give the same intformaiton
        - boxplots are better for showing mixing
YES - run it for longer - even if I can demonstrate mixing, is 10k steps enough?
        - probably not

## visualization tweaks
YES - standardize x-axis ticks for 1:1 comparison across random/enacted
- side-by-side the random/enacted plots and cutedges/percentage vs time plots
YES - make a flow chart that demonstrates data processing pipeline

## future work - don't really have time for any of these, I don't think
- bring in census data and do demographic analysis via ensemble (hispanic)
- operationalizing Hispanic majority, 2% of ideal population level, not splitting cities, and not changing from previous map for optimization via acceptance/rejection chain

definitely do one of these
- ei of demographic data
    - alternatively, CVR by precinct for the general election
    - packages: intro to pyei with colab
        pymc==5.17.0
        jaxlib==0.4.17
    - upload to colab notebook (or don't update your packages)
    - for independent voters, can you figure out who independent voters vote for?
    - currently i don't have the data
- census data
    - get a census shapefile
    - spatial join
    - shapefile for census blocks from tigerline
    - csv for demographic data, join it on
    - follow the steps from class, aggregate demo data from blocks to precincts

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
- https://thenevadaindependent.com/article/nobodys-redistricting-anything-this-weekend

## data and method
### data
- source and timeliness of data
- explain what a voter file is
    - briefly
- use of statewide election
    - population variance weirdness
    - turnout vs total population
    - whether commissioners were on the ballot
- trustworthiness of pipeline
### method - this can be brief
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
    - definitely all this stuff
    - what does my analysis do well
    - what does it not do well
    - how can it be improved
    - can i make conclusions/recommendations based on my model

## conclusion
- make an argument about partisan gerrymandering and optimizing for Hispanic majority districts
- appendix that includes outputs of `data_cleaning.py` and `main.py`
