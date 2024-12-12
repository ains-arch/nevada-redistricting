# Nevada's Clark County Commission Redistricting Analysis

In this report, we analyze partisan lean in Clark County.
We conduct an ensemble analysis of Nevada's 2021 Clark County Commission districting map with the 2024 election voter file and party registration data.
Then, we focus on the preferences of independent voters using the cast vote record for the 2024 General Election.

## Data sources
This ensemble uses the following data from the Clark County Elections Department:
1. The [precinct shapefile](https://www.clarkcountynv.gov/government/departments/geographic_info_systems/services/free_gis_data1.php#outer-6012) for Clark County after the 2021 redistricting.
1. The current (2024-12-02) [voter registration file](https://www.clarkcountynv.gov/government/departments/elections/reports_data_maps/voter_list_data_files.php) for Clark County.
1. A [database of all voters](https://www.clarkcountynv.gov/government/departments/elections/past_elections.php) in the 2024 General Election.
1. The [Cast Vote Record (CVR)](https://www.clarkcountynv.gov/government/departments/elections/) for the 2024 General Election.

This ensemble analysis is implemented using [Gerrychain](https://github.com/mggg/GerryChain) and focuses on partisan lean in County Commission seats, as demonstrated by the party affiliation of 2024 voters, and the ecological inference is implemented using [PyEI](https://github.com/mggg/ecological-inference) and the 2024 presidential election results.

## Data processing
This work is found in `data_cleaning.py`, and the output can be found in `data_cleaning.log`.

We process and merge voter registration, voter demographic data, precinct boundary data, and the cast votes to create a final dataset and shapefile containing aggregated voter registration counts by political party and aggregated votes for president for each precinct.

### Clean and Merge Voter Registration Data (`registration.csv` and `voters.csv`)
- Inputs:
  - `registration.csv`: The voter registration file. Data includes precinct, districts (Congress, Assembly, Senate, Commission), party registration, and registration number.  
  - `voters.csv`: The 2024 voters file. Data includes registration number and voting method.
- Steps:
  - Chunk-based processing: Read `registration.csv` in chunks of 100,000 rows to handle memory limitations.
  - Column selection: Select columns related to precinct, district assignments, party registration, and unique voter ID, and assign data types to decrease memory usage.
  - Merging: Join cleaned `registration.csv` and `voters.csv` using the unique voter ID to create a combined dataset.
- Output:
  - `registration_fixed.csv` and `voters_fixed.csv`: Intermediate cleaned datasets.  
  - `filtered_voters.csv`: Final merged voter data.

### Cleaning and Preparing Precinct Shapefile (`precinct_p.shp`):
- Input:
  - `precinct_p.shp`: Shapefile with boundary data for electoral precincts, including geometries and attributes like precinct ID.
- Steps:
  - Validation: Use `maup` to identify and fix invalid geometries.
  - Smart repair: Resolve overlaps and ensure the geometries are valid.
  - Deduplication: Identify and merge `MultiPolygon` districts.
  - Sanity check: Validate the dual graph connectivity.
- Output:  
  - `precinct_p_fixed.shp`: Cleaned precinct boundary shapefile.  
  - Diagnostic plots saved as `original_gdf.png`, `repaired.png`, and `merged.png`.
<img src=figs/original_gdf.png/>
<img src=figs/repaired.png/>
<img src=figs/merged.png/>

### Merging Voter Data with Precinct Boundaries:
- Inputs:  
  - `filtered_voters.csv`: Aggregated voter data.  
  - `precinct_p_fixed.shp`: Cleaned shapefile of precinct geometries.
- Steps:
  - Alignment check: Match precinct IDs across datasets and remove unmatched voters.
  - Aggregation: Group voter data by precinct and party registration to count the number of voters in each party per precinct.
  - Merging: Combine aggregated voter counts with precinct boundary data.
- Output:  
  - Final Shapefile (`aggregated_precincts.shp`): Include precinct boundaries, district assignments, and voter counts by party.  
  - Final CSV (`aggregated_precincts.csv`): Equivalent data without geometry.

### Processing Cast Vote Records (`cast_votes.csv`) ### 
- Input:
    - `cast_vote.csv`: Every ballot counted, with precinct ID.
- Steps:
    - Initial setup: Define helper function to extract and clean precinct IDs.
    - Vote records processing: Read `cast_votes.csv` in chunks, sum together third party candidates into an `other` column, clean precinct IDs.
    - Chunk aggregation: Sum votes for Harris, Trump, and other candidates by precinct in the chunk, and calculate total votes cast in each precinct in the chunk.
    - Final aggregation: Group aggregated votes by precinct, aggregating votes for all candidates and total vote count, all by precinct.
- Output:
    - `cast_vote_fixed.csv`: Number of votes cast for Harris, Trump, and other candidates by precinct.

### Merging with Precinct Boundaries ###
- Inputs:
    - `aggregated_precincts.shp`: Shapefile containing precinct boundaries, district assignments, and voter counts by party.
    - `cast_vote_fixed.csv`: Number of votes cast for Harris, Trump, and other candidates by precinct.
- Steps:
   - Merge the votes data with the aggregated precinct shapefile using a left join.
   - Fill missing vote counts in precincts with 0s.
- Outputs:
   - `final_precincts.csv`: all precinct attributes without geometries.
   - `final_precincts.shp`: containing precinct geometries and vote counts.

### Keys
- Between `registration.csv` and `voters.csv`: Unique voter ID (`REGISTRATION_NUM` and `idnumber`).  
- Between `filtered_voters.csv` and `precinct_p_fixed.shp`: Precinct ID (`PRECINCT` and `PREC`).  
- Between `cast_vote.csv` and `aggregated_precincts.shp`: `precinct` ID.

# Background
## How many districts?
The Clark County Commission has seven members, elected from seven districts, which were redistricted after the 2020 Census during the 2021 redistricting process.
The districts are made up of precincts, which also changed lines during the redistricting process.

<img src=figs/precincts.png/>

## Who draws them?
The county commission at the time of the 2021 redistricting process was seven Democrats.
The county hired Dave Heller, a national consultant, to redraw the maps.
There was some controversy around his hiring due to his background as a Democratic media consultant and campaign strategist.
The maps were redrawn with an eye toward creating a second Hispanic majority district, while staying within 2% of the ideal population level and keeping cities in the same district with minimal change from the previous map.
This analysis does not involve racial demographic data, and instead focuses on the partisan lean effects of this districting map.

## Response
There were no legal challenges.
Some community groups were concerned the new maps would pit minority groups against each other, as one of the majority-Hispanic districts is also the district with the biggest share of Black residents.
There were also concerns about a lack of community input, with only one presentation before commissioners voted to approve the map.
Additionally, demographic information was not available during that meeting.
However, generally there was support for increased Latino representation.

At the state level, the legislative map faced a legal challenge by Republicans after being signed into law by Gov. Steve Sisolak (D) and the Democratic-majority Legislature.
It was focused on partisan effects, alleging that the maps violated equal participation and opportunity by diluting representation for Republican and nonpartisan voters.
The case was blocked ahead of the 2022 election, and then dismissed.

## Setting
Clark County's registration distribution for active voters is:
| Party | Percentage |
|-|-|
| Democratic | 31.98%|
| Republican | 25.91%|
| Nonpartisan| 35.12%|
| Other      |  7.00%|

The results for the presidential race in the 2024 General Election are:
| Candidate | Percentage |
|--------|--------|
| Harris | 50.44% |
| Trump  | 47.81% |
| Other  |  1.75% |

Below is a set of maps that illustrate the partisan geography of Nevada, as represented by registered voters.
### Democratic 2024 voters
<img src=figs/dem_perc.png/>
<img src=figs/dem_perc_assembly.png/>
<img src=figs/dem_perc_senate.png/>
<img src=figs/dem_perc_congress.png/>
<img src=figs/dem_perc_commission.png/>

### Republican 2024 voters
<img src=figs/rep_perc.png/>
<img src=figs/rep_perc_assembly.png/>
<img src=figs/rep_perc_senate.png/>
<img src=figs/rep_perc_congress.png/>
<img src=figs/rep_perc_commission.png/>

### Non-partisan 2024 voters
Due to the easy of registration through the DMV, in which voters are not asked to choose a party affiliation, Nevada has a very high percentage of voters not registered with any party.
Here, we show the distribution of both non-partisan voters and voters registered with non-major parties.
<img src=figs/ind_perc.png/>
<img src=figs/ind_perc_assembly.png/>
<img src=figs/ind_perc_senate.png/>
<img src=figs/ind_perc_congress.png/>
<img src=figs/ind_perc_commission.png/>

# Findings
This work is found in `main.py`, and the output can be found in `main.log`.

This report includes data from an ensemble analysis of possible county commission district maps that could have been drawn after the 2020 census.
It was run for 100,000 steps.
This mixing time is sufficient, as can be shown in the similarity of the following boxplots and histograms, which begin their random walks either from the true enacted plan, or from a randomly generated plan.

The ensemble analysis focused on partisan lean as shown through the number of people who voted in the 2024 election registered with each major party, and with no party.
Likely due to the dispersion of party registration and the high percentage of non-partisan voters, our ensemble found no plans with even one Democrat or Republican majority commission district.
Instead, our histograms show plurality districts.

## Cutedges
<img src=figs/histogram-cutedges-enacted.png/>
<img src=figs/histogram-cutedges-random.png/>

## Democratic distribution
<img src=figs/histogram-democratic-plurality-enacted.png/>
<img src=figs/histogram-democratic-plurality-random.png/>

### Boxplots
<img src=figs/boxplot-democratic-enacted.png/>
<img src=figs/boxplot-democratic-random.png/>
While the top edge of our highest concentration district approached the 0.5 mark, no plans in this ensemble crossed it.

## Republican distribution
<img src=figs/histogram-republican-plurality-enacted.png/>
<img src=figs/histogram-republican-plurality-random.png/>

### Boxplots
<img src=figs/boxplot-republican-enacted.png/>
<img src=figs/boxplot-republican-random.png/>
There are fewer registered Republicans in Clark County.

## Non-partisan distribution
<img src=figs/histogram-independent-plurality-enacted.png/>
<img src=figs/histogram-independent-plurality-random.png/>
Despite being the deciding swing voters in elections, non-partisan voters are too dispersed to form a plurality district in our ensemble.

### Boxplots
<img src=figs/boxplot-independent-enacted.png/>
<img src=figs/boxplot-independent-random.png/>

## Ecological inference on independent voters in Clark County
In order to surmise the voting preference of independent (non-partisan and third party registered voters) in the 2024 presidential election, we run an ecological inference model.

<img src=figs/ei.png/>

Our posterior means for voting preferences among independents are 45.6% for Harris, 49.2% for Trump, and 5.2% for other candidates.

## Conclusion
While Clark County did vote for Harris overall, it was by a narrower margin than for Democrats in previous years, and Trump won the state at large.
At the county commission level, three of the four seats on the ballot were won by Democrats and one was won by a Republican.

This analysis looks at indirect ways of analyzing districts for gerrymandering, through imperfect metrics like party registration in a state dominated by the independent voice, and votes in a single race in a single election.
Additionally, statistical tools like ensemble analyses are blunt instruments that cannot fully model complex goals and settings.
Most notably, this analysis does not include any demographic information, and thus cannot contend with the stated intentions of the Clark County Commission in drawing a second Hispanic-majority district.
Thus, while our ensemble analysis may be a good sample of plans drawn with no goals but those operationalized in Gerrychain's implementation of recombination, it does not accurately traverse the space of possible plans in the mind of those drawing the map.
As such, conclusions drawn from our analysis should be limited in scope and open to better data, especially in the form of demographic analysis.

# Reproducing and running this code
All the code necessary to reproduce the results is included in this repository. To get
started, follow these steps to set up your development environment. You may need to install
Python 3.10.
```
$ python3.10 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip3 install -r requirements.txt
```
Sources for the data and the names that should be used for the files are listed in more detail in `data/sources.md`.

To exactly reproduce the results in this report, run:
```
$ python3 data_cleaning.py
$ python3 main.py --total_steps 10000
```
These are computationally intensive tasks, so taking steps to prevent canceled processes is
recommended.

# Sources and inspiration
The work done by The Nevada Independent on the effects of redistricting in Clark County was the original inspiration from this project.
You can read about their analysis [here](https://thenevadaindependent.com/article/analysis-how-redistricting-helped-nevada-democrats-but-not-enough-to-gain-supermajority) and [here](https://github.com/eneugebo/Redistricting/blob/main/README.md).

I additionally referred to [this article](https://nevadacurrent.com/2021/11/03/clark-county-approves-new-political-maps-including-2nd-hispanic-majority-district/) from The Nevada Current for historical and procedural information about the Clark County Commission's redistricting process.

I built off of a codebase I developed for a previous redistricting project hosted [here](https://github.com/ains-arch/colorado-redistricting).

Information about the legal challenges to the 2021 redistricting process at the state level from [The Nevada Independent](https://thenevadaindependent.com/article/gop-assemblyman-files-lawsuit-challenging-democrats-redistricting-plan) and [The American Redistricting Project](https://thearp.org/litigation/koenig-v-nevada/).

Exit polling from [CNN](https://www.cnn.com/election/2024/exit-polls/nevada/general/president/0).
