# Nevada's Clark County Commission Redistricting Analysis

In this report, we detail an ensemble analysis of Nevada's 2021 Clark County Commission districting map with the 2024 election voterfile and party registration data.

## Data sources
This ensemble uses the following data from the Clark County Elections Department:
1. The [precinct shapefile](https://www.clarkcountynv.gov/government/departments/geographic_info_systems/services/free_gis_data1.php#outer-6012) for Clark County after the 2021 redistricting.
1. The current (2024-12-02) [voter registration file](https://www.clarkcountynv.gov/government/departments/elections/reports_data_maps/voter_list_data_files.php) for Clark County.
1. A [database of all voters](https://www.clarkcountynv.gov/government/departments/elections/past_elections.php) in the 2024 General Election.

This ensemble analysis is implemented using Gerrychain and focuses on partisan lean in County Commission seats, as demonstrated by the party affiliation of 2024 voters.

## Data processing
This work is found in `data_cleaning.py`.

We process and merge voter registration, voter demographic data, and precinct boundary data to create a final dataset and shapefile containing aggregated voter counts by political party for each precinct.

### Cleaning and Merging Voter Registration Data (`registration.csv` and `voters.csv`)
- Inputs:
  - `registration.csv`: The voter registration file. Data includes precinct, districts (Congress, Assembly, Senate, Commission), party registration, and registration number.  
  - `voters.csv`: The 2024 voters file. Data includes registration number and voting method.
- Steps:
  - Chunk-based processing: Reads `registration.csv` in chunks of 100,000 rows to handle memory limitations.
  - Column selection: Selects columns related to precinct, district assignments, party registration, and unique voter ID, and assigns data types to decrease memory usage.
  - Merging: Joins cleaned `registration.csv` and `voters.csv` using the unique voter ID to create a combined dataset.
- Output:
  - `registration_fixed.csv` and `voters_fixed.csv`: Intermediate cleaned datasets.  
  - `filtered_voters.csv`: Final merged voter data.

### Cleaning and Preparing Precinct Shapefile (`precinct_p.shp`):
- Input:  
  - `precinct_p.shp`: A shapefile with boundary data for electoral precincts, including geometries and attributes like precinct ID.
- Steps:
  - Validation: Uses `maup` to identify and fix invalid geometries.
  - Smart repair: Resolve overlaps and ensure the geometries are valid.
  - Deduplication: Identifies and merges `MultiPolygon` districts.
  - Sanity check: Validates the dual graph connectivity.
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
  - Alignment check: Matches precinct IDs across datasets and removes unmatched voters.
  - Aggregation: Groups voter data by precinct and party registration to count the number of voters in each party per precinct.
  - Merging: Combines aggregated voter counts with precinct boundary data.
- Output:  
  - Final Shapefile (`aggregated_precincts.shp`): Includes precinct boundaries, district assignments, and voter counts by party.  
  - Final CSV (`aggregated_precincts.csv`): Equivalent data without geometry.

### Keys
- Between `registration.csv` and `voters.csv`: Unique voter ID (`REGISTRATION_NUM` and `idnumber`).  
- Between `filtered_voters.csv` and `precinct_p_fixed.shp`: Precinct ID (`PRECINCT` and `PREC`).  

# Background
## How many districts?
The Clark County Commission has seven members, elected from seven districts, which were redistricted after the 2020 Census during the 2021 redistricting process.
The districts are made up of precincts, which also changed lines during the redistricting process.

<img src=figs/precincts.png/>

## Who draws them?
The county commission at the time of the 2021 redistricting process was seven Democrats.
The county hired Dave Heller, a national consultant, to redraw the maps.
The maps were redrawn with an eye toward creating a second Hispanic majority district, while staying within 2% of the ideal population level and keeping cities in the same district with minimal change from the previous map.
This analysis does not involve racial demographic data, and instead focuses on the partisan lean effects of this districting map.

## Response
There were no legal challenges.
Some community groups were concerned the new maps would pit minority groups against each other, as one of the majority-Hispanic districts is also the district with the biggest share of Black residents.
There were also concerns about a lack of community input, with only one presentation before commissioners voted to approve the map.
Additionally, demographic information was not available during that meeting.
However, generally there was support for increased Latino representation.

## Setting
Below is a set of maps that illustrate the partisan geography of Nevada.
### Democratic 2024 voters
<img src=figs/dem_perc.png/>
<img src=figs/dem_perc_assembly.png/>
<img src=figs/dem_perc_senate.png/>
<img src=figs/dem_perc_congress.png/>
<img src=figs/dem_perc_commission.png/>

### Republican 2024 voters
<img src=figs/np_perc.png/>
<img src=figs/np_perc_assembly.png/>
<img src=figs/np_perc_senate.png/>
<img src=figs/np_perc_congress.png/>
<img src=figs/np_perc_commission.png/>

Due to the easy of registration through the DMV, in which voters are not asked to choose a party affiliation, Nevada has a very high percentage of non-partisan voters.
### Independent 2024 voters
<img src=figs/rep_perc.png/>
<img src=figs/rep_perc_assembly.png/>
<img src=figs/rep_perc_senate.png/>
<img src=figs/rep_perc_congress.png/>
<img src=figs/rep_perc_commission.png/>

# Findings
This report includes data from an ensemble analysis of possible county commission district maps that could have been drawn after the 2020 census.
It was run for 10,000 steps.
This mixing time is sufficient, as can be shown in the similarity of the following histograms, which begin their random walks either from the true enacted plan, or from a randomly generated plan.

The ensemble analysis focused on partisan lean as shown through the number of people who voted in the 2024 election registered with each major party, and with no party.

## Cutedges
<img src=figs/histogram-cutedges-enacted.png/>
<img src=figs/histogram-cutedges-random.png/>

## Democratic distribution
<img src=figs/histogram-democratic-majority-enacted.png/>
<img src=figs/histogram-democratic-majority-random.png/>
<img src=figs/histogram-democratic-plurality-enacted.png/>
<img src=figs/histogram-democratic-plurality-random.png/>

### Boxplots
<img src=figs/boxplot-democratic-enacted.png/>
<img src=figs/boxplot-democratic-random.png/>

## Republican distribution
<img src=figs/histogram-independent-majority-enacted.png/>
<img src=figs/histogram-independent-majority-random.png/>
<img src=figs/histogram-independent-plurality-enacted.png/>
<img src=figs/histogram-independent-plurality-random.png/>

### Boxplots
<img src=figs/boxplot-republican-enacted.png/>
<img src=figs/boxplot-republican-random.png/>


## Independent distribution
<img src=figs/histogram-republican-majority-enacted.png/>
<img src=figs/histogram-republican-majority-random.png/>
<img src=figs/histogram-republican-plurality-enacted.png/>
<img src=figs/histogram-republican-plurality-random.png/>

### Boxplots
<img src=figs/boxplot-independent-enacted.png/>
<img src=figs/boxplot-independent-random.png/>

## Future work
Further analysis of the demographic makeup of the districts, with a focus on Hispanic and other racial demographic outcomes would lend more information about whether the goals of the redistricting process were successful. Additionally, comparing to the previous plan would allow a wider picture of whether redistricing favored one party over the other.

# Reproducing and running this code
All the code and data necessary to reproduce the results is included in this repository. To get
started, follow these steps to set up your development environment. You may need to install
Python 3.10.
```
$ python3.10 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip3 install -r requirements.txt
```

To exactly reproduce the results in this report, run:
```
$ python3 main.py --total_steps 10000
```
This is a computationally intensive task, so taking steps to prevent canceled processes is
recommended.

# Sources and inspiration
The work done by The Nevada Independent on the effects of redistricting in Clark County was the original inspiration from this project.
You can read about their analysis [here](https://thenevadaindependent.com/article/analysis-how-redistricting-helped-nevada-democrats-but-not-enough-to-gain-supermajority) and [here](https://github.com/eneugebo/Redistricting/blob/main/README.md).

I additionally referred to [this article](https://nevadacurrent.com/2021/11/03/clark-county-approves-new-political-maps-including-2nd-hispanic-majority-district/) from The Nevada Current for historical and procedural information about the Clark County Commission's redistricting process.

I built off of a codebase I developed for a previous redistricting project hosted [here](https://github.com/ains-arch/colorado-redistricting).
