from functools import partial
import warnings
import argparse
import random
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pymc as pm
from gerrychain import Graph, Partition, GeographicPartition, MarkovChain, constraints
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from pyei.two_by_two import TwoByTwoEI
from pyei.goodmans_er import GoodmansER
from pyei.goodmans_er import GoodmansERBayes
from pyei.r_by_c import RowByColumnEI
from pyei.io_utils import to_netcdf
from pyei.io_utils import from_netcdf

warnings.filterwarnings("ignore") # Suppress warnings about NA values

# To reproduce results
import random
random.seed(89134)

# Set up argument parser
parser = argparse.ArgumentParser(description="Run random walk with a specified number of steps.")
parser.add_argument(
  '--total_steps', 
  type=int, 
  default=100, 
  help='The total number of steps for the random walk (default: 100)'
)

# Parse the arguments
args = parser.parse_args()

# Use the total_steps argument in code
total_steps = args.total_steps

# Set global matplotlib dpi for high quality plots
plt.rcParams['savefig.dpi'] = 300

# Load the data
shapefile_path = "data/final_precincts.shp"
print("loading shapefile")
gdf = gpd.read_file(shapefile_path)
print("shapefile loaded")
graph = Graph.from_geodataframe(gdf)
if not nx.is_connected(graph):
    raise ValueError("Graph is not connected")
print("gdf:\n", gdf.columns)

# Define a helper function for plotting
def plot_map(gdf, column, cmap, title, output_path, level_column=None):
    """
    Plots a choropleth map of a given column in a GeoDataFrame with district-level outlines.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing precinct geometries and data.
        column (str): The column to visualize.
        cmap (str): The colormap for the data.
        title (str): The title of the plot.
        output_path (str): The path to save the output image.
        level_column (str, optional): The column to group by for district outlines (e.g., "senate").
    """
    plt.figure(figsize=(10, 8))

    # Plot the precinct-level data
    ax = gdf.plot(
        column=column,
        cmap=cmap,
        missing_kwds={"color": "gray"},
        vmin=0, vmax=1,
        legend=False,
        alpha=0.8  # Slight transparency for better outline visibility
    )
    
    # Overlay district boundaries if level_column is specified
    if level_column:
        districts_gdf = gdf.dissolve(by=level_column)  # Aggregate geometries by level
        districts_gdf.boundary.plot(ax=ax, color="black", linewidth=0.2)
    else:
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.2)

    plt.axis('off')
    plt.title(title, fontsize=14)
    
    # Create a colorbar
    if cmap:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.01, shrink=0.7)
        cbar.set_label('Percentage', fontsize=12)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

# Plot precinct shapefile
print("plotting shapefile")
plot_map(gdf, None, None, "Clark County Precincts", "figs/precincts.png")

### ANALYZING PARTY ###

# Calculate total voters per precinct
gdf['total_voters'] = gdf[['DEM', 'FOR','GRN', 'IAP', 'LPN', 'NAT', 'NFP', 'NLN', 'NP', 'NTP',
                           'OTH', 'REF','REP', 'TPN', 'WSP']].sum(axis=1)

## Voter registration ##
# Calculate percentage columns for each party
gdf['dem_perc'] = gdf['DEM'] / gdf['total_voters']
gdf['rep_perc'] = gdf['REP'] / gdf['total_voters']
gdf['np_perc'] = gdf['NP'] / gdf['total_voters']
gdf['ind_perc'] = (gdf['total_voters'] - gdf['DEM'] - gdf['REP']) / gdf['total_voters']
gdf = gdf.fillna(0)

# Plot maps for each party
plot_map(gdf, 'dem_perc', 'Blues',
         'Democratic Percentage by Precinct',
         'figs/dem_perc.png')
plot_map(gdf, 'rep_perc', 'Reds',
         'Republican Percentage by Precinct',
         'figs/rep_perc.png')
plot_map(gdf, 'np_perc', 'viridis', 
         'Non-Partisan Percentage by Precinct',
         'figs/np_perc.png')
plot_map(gdf, 'ind_perc', 'YlOrBr', 
         'Independents Percentage by Precinct',
         'figs/ind_perc.png')

# Sanity checks #
# Reload graph
graph = Graph.from_geodataframe(gdf)
if not nx.is_connected(graph):
    raise ValueError("Graph is no longer connected")

# Verify percentages sum to 1 for precincts with voters
# Create a mask for precincts with non-zero total voters
non_zero_voters_mask = gdf['total_voters'] > 0

# Check percentages for precincts with voters
percentage_check = np.isclose(
    gdf.loc[non_zero_voters_mask, 'dem_perc'] + 
    gdf.loc[non_zero_voters_mask, 'rep_perc'] + 
    gdf.loc[non_zero_voters_mask, 'ind_perc'], 
    1.0, 
    atol=1e-10  # absolute tolerance to account for floating-point imprecision
)

# If any rows fail the check, print them out
problematic_rows = gdf.loc[non_zero_voters_mask][~percentage_check]
if not problematic_rows.empty:
    print("Rows where percentages do not sum to 1:")
    print(problematic_rows[['precinct', 'dem_perc', 'rep_perc', 'ind_perc', 'total_voters']])
    print(f"Number of problematic rows: {len(problematic_rows)}")
    
# Raise an error if there are any problematic rows
if not problematic_rows.empty:
    raise ValueError("Voter percentages do not sum to 1 in all precincts")

# Print populations and percentages
dempop = sum(graph.nodes()[v]['DEM'] for v in graph.nodes())
print(f"Democratic voters: {dempop}")
reppop = sum(graph.nodes()[v]['REP'] for v in graph.nodes())
print(f"Republican voters: {reppop}")
nppop = sum(graph.nodes()[v]['NP'] for v in graph.nodes())
print(f"Nevada independents: {nppop}")
totpop = sum(graph.nodes()[v]['total_voters'] for v in graph.nodes())
print(f"Total voters: {totpop}")
print(f"Democratic percentage: {dempop/totpop:.2%}")
print(f"Republican percentage: {reppop/totpop:.2%}")
print(f"Nevada independents percentage: {nppop/totpop:.2%}")

## Votes for President in 2024 ##
# Calculate vote percentage columns for presidential candidates
gdf['harris_perc'] = gdf['harris'] / gdf['total_vote']
gdf['trump_perc'] = gdf['trump'] / gdf['total_vote']
gdf['other_perc'] = gdf['other'] / gdf['total_vote']
gdf = gdf.fillna(0)

# Print vote tallies and percentages

# Plot maps for each party
plot_map(gdf, 'harris_perc', 'Blues',
         'Harris Percentage by Precinct',
         'figs/harris_perc.png')
plot_map(gdf, 'trump_perc', 'Reds',
         'Trump Percentage by Precinct',
         'figs/trump_perc.png')
plot_map(gdf, 'other_perc', 'YlOrBr', 
         'Other by Precinct',
         'figs/other_perc.png')

# Sanity checks #
# Reload graph
graph = Graph.from_geodataframe(gdf)
if not nx.is_connected(graph):
    raise ValueError("Graph is no longer connected")

# Verify percentages sum to 1 for precincts with votes
# Create a mask for precincts with non-zero total votes
non_zero_votes_mask = gdf['total_vote'] > 0

# Check percentages for precincts with votes
percentage_check = np.isclose(
    gdf.loc[non_zero_votes_mask, 'harris_perc'] + 
    gdf.loc[non_zero_votes_mask, 'trump_perc'] + 
    gdf.loc[non_zero_votes_mask, 'other_perc'], 
    1.0, 
    atol=1e-10  # absolute tolerance to account for floating-point imprecision
)

# If any rows fail the check, print them out
problematic_rows = gdf.loc[non_zero_votes_mask][~percentage_check]
if not problematic_rows.empty:
    print("Rows where percentages do not sum to 1:")
    print(problematic_rows[['precinct', 'harris_perc', 'trump_perc', 'other_perc', 'total_voters']])
    print(f"Number of problematic rows: {len(problematic_rows)}")
    
# Raise an error if there are any problematic rows
if not problematic_rows.empty:
    raise ValueError("Candidate percentages do not sum to 1 in all precincts")

### ANALYZING DISTRICTS ###
def calculate_percentages(district_stats, level):
    """
    Adds percentage columns for each party to the district stats DataFrame.
    
    Parameters:
        district_stats (DataFrame): Aggregated district statistics.
        level (str): Level of analysis (e.g., 'congress', 'assembly').
    
    Returns:
        DataFrame: Updated district stats with percentage columns.
    """
    print(district_stats)
    district_stats[f'dem_perc_{level}'] = district_stats['DEM'] / district_stats['total_voters']
    district_stats[f'rep_perc_{level}'] = district_stats['REP'] / district_stats['total_voters']
    district_stats[f'np_perc_{level}'] = district_stats['NP'] / district_stats['total_voters']
    return district_stats


def calculate_majorities_and_pluralities(district_stats, level, party_col, perc_col):
    """
    Calculates plurality and majority counts for a specific party.
    
    Parameters:
        district_stats (DataFrame): Aggregated district statistics.
        level (str): Level of analysis (e.g., 'congress', 'assembly').
        party_col (str): Column name for the party's raw vote count.
        perc_col (str): Column name for the party's vote percentage.
    
    Returns:
        tuple: (plurality_count, majority_count)
    """
    plurality_count = (
        (district_stats[party_col] >= district_stats['DEM']) &
        (district_stats[party_col] >= district_stats['REP']) &
        (district_stats[party_col] >= district_stats['NP'])
    ).sum()
    majority_count = (district_stats[perc_col] >= 0.5).sum()
    return plurality_count, majority_count

def analyze_districts(gdf, level, graph, cut_edges, summary_df=None):
    """
    Analyzes districts at a given level (e.g., 'congress', 'assembly', 'senate').

    Parameters:
        gdf (GeoDataFrame): The input geodataframe containing voter data.
        level (str): The column name representing the level of analysis (e.g., 'congress').
        graph: Graph object required for GeographicPartition.
        cut_edges: Function or updater for calculating cut edges.
        summary_df (DataFrame): Optional DataFrame to append results for all levels.
    
    Returns:
        tuple: (partition, GeoDataFrame, Graph, DataFrame)
            partition: Updated partition.
            gdf: GeoDataFrame with percentage columns for the level.
            graph: Updated graph object.
            summary_df: DataFrame containing the calculated summary statistics.
    """
    print(f"Turning 2024 {level} map into a partition...")
    partition = GeographicPartition(
        graph,
        assignment=level,
        updaters={"cutedges": cut_edges},
    )

    print(f"Running aggregation into {level} districts...")
    # Aggregate data using groupby
    district_stats = gdf.groupby(level).agg({
        'total_voters': 'sum',
        'DEM': 'sum',
        'REP': 'sum',
        'NP': 'sum'
    }).reset_index()

    # Add percentage columns
    district_stats = calculate_percentages(district_stats, level)

    # Calculate plurality and majority counts for each party
    parties = [
        ("Democratic", "DEM", f'dem_perc_{level}'),
        ("Republican", "REP", f'rep_perc_{level}'),
        ("Independent", "NP", f'np_perc_{level}'),
    ]
    
    plurality_counts = {}
    majority_counts = {}
    
    for party_name, party_col, perc_col in parties:
        plurality, majority = calculate_majorities_and_pluralities(district_stats, level, party_col, perc_col)
        plurality_counts[party_name] = plurality
        majority_counts[party_name] = majority
        print(f"  Number of {party_name} plurality districts: {plurality}")
        print(f"  Number of {party_name} majority districts: {majority}")

    # Calculate number of cutedges
    cutedges = len(partition['cutedges'])
    print(f"  Number of cutedges: {cutedges}")

    # Merge the aggregated data back into the GeoDataFrame
    percentage_cols = [f'dem_perc_{level}', f'rep_perc_{level}', f'np_perc_{level}']
    gdf = gdf.merge(district_stats[[level] + percentage_cols], on=level, how='left')

    # Update summary DataFrame
    if summary_df is None:
        summary_df = pd.DataFrame()

    summary_entry = pd.DataFrame([{
        "level": level,
        "cutedges": cutedges,
        "dem_perc": district_stats[f"dem_perc_{level}"],
        "rep_perc": district_stats[f"rep_perc_{level}"],
        "np_perc": district_stats[f"np_perc_{level}"],
        "dem_plurality": plurality_counts["Democratic"],
        "rep_plurality": plurality_counts["Republican"],
        "np_plurality": plurality_counts["Independent"],
        "dem_majority": majority_counts["Democratic"],
        "rep_majority": majority_counts["Republican"],
        "np_majority": majority_counts["Independent"],
        }])
    summary_df = pd.concat([summary_df, summary_entry], ignore_index=True)

    # Plot maps
    plot_map(gdf, parties[0][2], 'Blues',
             f'Democratic Percentage by {level.capitalize()} District',
             f'figs/dem_perc_{level}.png', level)

    plot_map(gdf, parties[1][2], 'Reds',
             f'Republican Percentage by {level.capitalize()} District',
             f'figs/rep_perc_{level}.png', level)

    plot_map(gdf, parties[2][2], 'viridis',
             f'Independent Percentage by {level.capitalize()} District',
             f'figs/np_perc_{level}.png', level)

    # Reloading graph
    graph = Graph.from_geodataframe(gdf)
    if not nx.is_connected(graph):
        raise ValueError("Graph is no longer connected")

    return partition, gdf, graph, summary_df

# Initialize summary DataFrame
summary_df = None

# Analyze districts at different levels
for level in ['congress', 'assembly', 'senate', 'commission']:
    partition, gdf, graph, summary_df = analyze_districts(gdf, level, graph, cut_edges, summary_df)

# Access the summary data for specific levels or statistics
summary_df.to_csv("data/summary.csv", index=False)

print("Enacted maps data manipulation complete.\n")
print("gdf:\n", gdf.columns)

### ECOLOGICAL INFERENCE: RxC ###

# Create a new GeoDataFrame with only rows that have both total_vote and total_voters > 0
# Can't do EI on precincts that have no votes in them
ei_gdf = gdf[(gdf['total_vote'] > 0) & (gdf['total_voters'] > 0)].copy()

# Format the data
group_fractions = np.array(ei_gdf[['dem_perc', 'rep_perc', 'ind_perc']]).T
votes_fractions = np.array(ei_gdf[['harris_perc', 'trump_perc', 'other_perc']]).T
precinct_pops = np.array(ei_gdf['total_vote']).astype(int)
candidate_names = ["Harris", "Trump", "Other"]
demographic_group_names = ["Democrat", "Republican", "Independent"]

# Fitting a first model
ei = RowByColumnEI(model_name='multinomial-dirichlet')

# Fit the model
ei.fit(group_fractions,
       votes_fractions,
       precinct_pops,
       demographic_group_names=demographic_group_names,
       candidate_names=candidate_names,
       draws=1200,
       tune=3000,
       chains=4
)

# Save EI ouptut
to_netcdf(ei, 'data/ei.netcdf')

# Import EI output from previous run
# ei = from_netcdf('data/ei.netcdf')

# Generate a simple report to summarize the results
print(ei.summary())

plt.figure()
ei.plot()
plt.tight_layout()
plt.subplots_adjust(right=0.80)
plt.savefig("figs/ei.png")

### ENSEMBLE ANALYSIS ###

# Make an initial districting plan using recursive_tree_part
print("Setting up ensemble..")
NUM_DIST = len(gdf['commission'].unique()) # it's seven, by the way
ideal_pop = totpop/NUM_DIST
# This is really high because voter turnout varied a ton between commission districts
# If you're curious, the real population variance in the districts is 1-316k/364k = 0.132
# Which is still high, but less so? There also isn't a 1:1 map of pop to turnout in order
# A lot of things affect why people turn out in their districts, and not all of the commissioners
# are up for reelection each cycle, so I am just going to let it be.
POP_TOLERANCE = 0.33

def save_histogram(data, enacted_value, party, metric, color, flag, output_dir="figs"):
    """
    Saves a histogram for the ensemble data with appropriate styling.
    Parameters:
        data (list): Ensemble data for the metric (e.g., plurality or majority).
        enacted_value (int): The value of the metric in the enacted plan.
        party (str): The political party (e.g., "Democratic", "Republican", "Independent").
        metric (str): The metric type (e.g., "plurality", "majority").
        color (str): The color for the histogram (e.g., "blue", "red", "purple").
        flag (str): "random" or "enacted" based on start of ensemble.
        output_dir (str): Directory to save the plots.
    """
    min_value = min(data)
    max_value = max(data)
    bins = np.arange(min_value - 0.5, max_value + 1.5, 1)  # Integer bins
    if metric:
        ticks = np.arange(min_value, max_value + 1, 1)
    plt.figure()
    plt.hist(data, bins=bins, edgecolor='black', color=color)
    if metric:
        plt.xticks(ticks)
    else:
        plt.xlim(200, 425)  # Set x-axis for no metric
    plt.xlabel("Districts", fontsize=12)
    plt.ylabel("Ensembles", fontsize=12)
    if metric:
        plt.title(f"{party} {metric.capitalize()} Districts from {flag.capitalize()} Start", fontsize=14)
        plt.axvline(x=enacted_value, color='orange', linestyle='--', linewidth=2, 
                    label=f"Enacted plan = {enacted_value}")
    else:
        plt.title(f"{party} from {flag.capitalize()} Start", fontsize=14)
        plt.axvline(x=enacted_value, color='orange', linestyle='--', linewidth=2, 
                    label=f"Enacted plan = {enacted_value}")
    plt.legend()
    if metric:
        plt.savefig(f"{output_dir}/histogram-{party.lower()}-{metric}-{flag}.png")
        plt.close()
        print(f"Saved {output_dir}/histogram-{party.lower()}-{metric}-{flag}.png")
    else:
        plt.savefig(f"{output_dir}/histogram-{party.lower()}-{flag}.png")
        plt.close()
        print(f"Saved {output_dir}/histogram-{party.lower()}-{flag}.png")

def save_boxplot(data, enacted_values, party, color, flag, output_dir="figs"):
    """
    Saves a boxplot for party percentages by district at a given level.
    Parameters:
        data (list of lists): Ensemble data for the party percentages across districts.
        enacted_values (list): Enacted plan percentages for the party, one per district.
        party (str): The political party (e.g., "Democratic", "Republican", "Independent").
        color (str): The color for the boxplot (e.g., "blue", "red", "purple").
        flag (str): "random" or "enacted" based on start of ensemble.
        output_dir (str): Directory to save the plots.
    """
    a = np.array(data)
    # Sort enacted values for better visualization
    enacted_values_sorted = sorted(enacted_values)
    # Create the boxplot
    plt.figure()
    plt.boxplot(a, patch_artist=True, 
                boxprops=dict(facecolor=color, color='black'),
                medianprops=dict(color='goldenrod', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1),
                capprops=dict(color='black', linewidth=1),
                zorder=1)
    # Overlay the enacted plan as a scatter plot
    plt.scatter(x=range(1, len(enacted_values_sorted) + 1), y=enacted_values_sorted, 
                color="goldenrod", label="Enacted plan", zorder=2)
    # Add title, labels, and legend
    plt.xlabel("Districts", fontsize=12)
    plt.ylabel(f"{party} Percentage", fontsize=12)
    plt.suptitle(f"{party} Percentage in Commission Districts", fontsize=14)
    plt.title(f"from {flag.capitalize()} Start")
    plt.legend()
    # Save the boxplot
    filename = f"{output_dir}/boxplot-{party.lower()}-{flag}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def run_random_walk(enacted = True):
    if enacted:
        flag = "enacted"
        initial_plan = "commission"
    else:
        flag = "random"
        initial_plan = recursive_tree_part(graph,
                                           range(NUM_DIST),
                                           ideal_pop,
                                           'total_voters',
                                           POP_TOLERANCE,
                                           10)

    # Get lat/lon from shapefile geometry to be fed into nx.draw
    node_locations = {
        v: (
            gdf.loc[v, 'geometry'].centroid.x,  # Longitude (x-coordinate)
            gdf.loc[v, 'geometry'].centroid.y   # Latitude (y-coordinate)
        )
        for v in graph.nodes()
    }
    
    # Set up partition object
    initial_partition = Partition(
        graph, # dual graph
        assignment = initial_plan, # initial districting plan
        updaters = {
            "cutedges": cut_edges, 
            "totpop": Tally("total_voters", alias = "totpop"), 
            "dempop": Tally("DEM", alias = "dempop"),
            "reppop": Tally("REP", alias = "reppop"),
            "nppop": Tally("NP", alias = "nppop"),
        }
    )
    
    # Set up random walk
    rw_proposal = partial(recom,                   # How you choose a next districting plan
                          pop_col = "total_voters",# What data describes population
                          pop_target = ideal_pop,  # Target/ideal population is for each district
                          epsilon = POP_TOLERANCE, # How far from ideal population you can deviate
                          node_repeats = 1         # Number of times to repeat bipartition.
                          )
    
    # Set up population constraint
    population_constraint = constraints.within_percent_of_ideal_population(
            initial_partition,
            POP_TOLERANCE,
            pop_key = "totpop"
            )
    
    # Set up the chain
    our_random_walk = MarkovChain(
            proposal = rw_proposal,
            constraints = [population_constraint],
            accept = always_accept, # accepts every proposed plan that meets population criteria
            initial_state = initial_partition,
            total_steps = total_steps
            )
    
    # Run the random walk
    print(f"Running random walk from {flag} start...")
    cutedge_ensemble = []
    d_plu_ensemble = []
    d_maj_ensemble = []
    dempop = []
    r_plu_ensemble = []
    r_maj_ensemble = []
    reppop = []
    np_plu_ensemble = []
    np_maj_ensemble = []
    nppop = []
    
    for part in our_random_walk:
        # Add cutedges to cutedges ensemble
        cutedge_ensemble.append(len(part["cutedges"]))
    
        d_plu = 0
        r_plu = 0
        np_plu = 0
        d_maj = 0
        r_maj = 0
        np_maj = 0
        dempop_this_step = []
        reppop_this_step = []
        nppop_this_step = []

        for district in part.parts:
            d_perc = part["dempop"][district]/part["totpop"][district]
            dempop_this_step.append(d_perc)
            r_perc = part["reppop"][district]/part["totpop"][district]
            reppop_this_step.append(r_perc)
            np_perc = part["nppop"][district]/part["totpop"][district]
            nppop_this_step.append(np_perc)

            # plurality districts
            if d_perc >= r_perc and d_perc >= np_perc:
                d_plu += 1
            if r_perc >= d_perc and r_perc >= np_perc:
                r_plu += 1
            if np_perc >= d_perc and np_perc >= r_perc:
                np_plu += 1

            # majority districts
            if d_perc >= 0.5:
                d_maj += 1
            if r_perc >= 0.5:
                r_maj += 1
            if np_perc >= 0.5:
                np_maj += 1

        d_plu_ensemble.append(d_plu)
        d_maj_ensemble.append(d_maj)
        dempop_this_step.sort()
        dempop.append(dempop_this_step)

        r_plu_ensemble.append(r_plu)
        r_maj_ensemble.append(r_maj)
        reppop_this_step.sort()
        reppop.append(reppop_this_step)

        np_plu_ensemble.append(np_plu)
        np_maj_ensemble.append(np_maj)
        nppop_this_step.sort()
        nppop.append(nppop_this_step)
    
    print("Random walk complete.\n")

    # Color mapping for parties
    party_colors = {"Democratic": "blue", "Republican": "red", "Independent": "purple"}

    # Shorthand
    party_short = {"Democratic": "dem", "Republican": "rep", "Independent": "np"}
    
    # Map summary_df columns to metrics
    metrics_mapping = {
        "plurality": {"Democratic": "dem_plurality", "Republican": "rep_plurality", "Independent": "np_plurality"},
        "majority": {"Democratic": "dem_majority", "Republican": "rep_majority", "Independent": "np_majority"}
    }
    
    # Party data for boxplots
    district_ensemble_data = {
        "Democratic": dempop,
        "Republican": reppop,
        "Independent": nppop
    }
    
    # Save the histogram for cutedges
    save_histogram(cutedge_ensemble, 
                   summary_df.loc[summary_df['level'] == 'commission', "cutedges"].values[0],
                   "Cutedges", None, "brown", flag)
    
    # Histograms
    for metric, party_columns in metrics_mapping.items():
        for party, column in party_columns.items():
            enacted_value = summary_df.loc[summary_df['level'] == 'commission', column].values[0]
            # Select the appropriate ensemble data
            data = d_plu_ensemble if party == "Democratic" and metric == "plurality" else \
                   d_maj_ensemble if party == "Democratic" and metric == "majority" else \
                   r_plu_ensemble if party == "Republican" and metric == "plurality" else \
                   r_maj_ensemble if party == "Republican" and metric == "majority" else \
                   np_plu_ensemble if party == "Independent" and metric == "plurality" else \
                   np_maj_ensemble
    
            # Save the histogram
            save_histogram(data, enacted_value, party, metric, party_colors[party], flag)

    # Boxplots
    for party, data in district_ensemble_data.items():
        # Get enacted values for the current party
        enacted_values = summary_df.loc[summary_df['level'] == "commission", f"{party_short[party]}_perc"].values[0]
        
        # Save the boxplot
        save_boxplot(data, enacted_values, party, party_colors[party], flag)

run_random_walk()
run_random_walk(enacted = False)
