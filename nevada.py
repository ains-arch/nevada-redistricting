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
from gerrychain import Graph, Partition, GeographicPartition, MarkovChain, constraints
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept

# warnings.filterwarnings("ignore") # Suppress warnings about NA values

# To reproduce results
# import random
# random.seed(188923)

# Set up argument parser
# parser = argparse.ArgumentParser(description="Run random walk with a specified number of steps.")
# parser.add_argument(
    # '--total_steps', 
    # type=int, 
    # default=100, 
    # help='The total number of steps for the random walk (default: 100)'
# )
total_steps = 100

# Parse the arguments
# args = parser.parse_args()

# Use the total_steps argument in your code
# total_steps = args.total_steps

# Set global matplotlib dpi for high quality plots
# plt.rcParams['savefig.dpi'] = 300

# Load the data
shapefile_path = "data/aggregated_precincts.shp"
print("loading shapefile")
gdf = gpd.read_file(shapefile_path)
print("shapefile loaded")
print("loading graph")
graph = Graph.from_geodataframe(gdf)
print("graph loaded")
print(f"Is the dual graph connected? {nx.is_connected(graph)}")
print("gdf:\n", gdf.columns)

# Define a helper function for plotting
def plot_map(gdf, column, cmap, title, output_path):
    plt.figure(figsize=(10, 8))
    ax = gdf.plot(
        column=column,
        cmap=cmap,
        missing_kwds={"color": "gray"},
        vmin=0, vmax=1,
        legend=False
    )
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

print("analyzing party")

# Calculate total voters per precinct
gdf['total_voters'] = gdf[['DEM', 'FOR','GRN', 'IAP', 'LPN', 'NAT', 'NFP', 'NLN', 'NP', 'NTP',
                           'OTH', 'REF','REP', 'TPN', 'WSP']].sum(axis=1)

# Calculate percentage columns for each party
gdf['dem_perc'] = gdf['DEM'] / gdf['total_voters']
gdf['rep_perc'] = gdf['REP'] / gdf['total_voters']
gdf['np_perc'] = gdf['NP'] / gdf['total_voters']
gdf = gdf.fillna(0)

# Plot maps for each party
plot_map(gdf, 'dem_perc', 'Blues',
         'Democratic Registration Percentage by Precinct',
         'figs/dem_perc.png')
plot_map(gdf, 'rep_perc', 'Reds',
         'Republican Registration Percentage by Precinct',
         'figs/rep_perc.png')
plot_map(gdf, 'np_perc', 'viridis', 
         'Non-Partisan Registration Percentage by Precinct',
         'figs/np_perc.png')

# Reloading graph
print("Loading updated graph...")
graph = Graph.from_geodataframe(gdf)
print("graph reloaded")
print("Updated graph loaded.")
print(f"Information at nodes: {graph.nodes()[0].keys()}")
print(f"Is the dual graph still connected? {nx.is_connected(graph)}")

# Print populations
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

print("analyzing districts")

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


def analyze_districts(gdf, level, graph, cut_edges):
    """
    Analyzes districts at a given level (e.g., 'congress', 'assembly', 'senate').

    Parameters:
        gdf (GeoDataFrame): The input geodataframe containing voter data.
        level (str): The column name representing the level of analysis (e.g., 'congress').
        graph: Graph object required for GeographicPartition.
        cut_edges: Function or updater for calculating cut edges.
    
    Returns:
        GeoDataFrame: The original GeoDataFrame with additional percentage columns for the level.
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
        ("Nevada Independent", "NP", f'np_perc_{level}'),
    ]
    
    for party_name, party_col, perc_col in parties:
        plurality, majority = calculate_majorities_and_pluralities(district_stats, level, party_col, perc_col)
        print(f"  Number of {party_name} plurality districts: {plurality}")
        print(f"  Number of {party_name} majority districts: {majority}")

    # Calculate number of cutedges
    cutedges = len(partition['cutedges'])
    print(f"  Number of cutedges: {cutedges}")

    # Merge the aggregated data back into the GeoDataFrame
    percentage_cols = [f'dem_perc_{level}', f'rep_perc_{level}', f'np_perc_{level}']
    gdf = gdf.merge(district_stats[[level] + percentage_cols], on=level, how='left')

    # TODO: change first letter of {level} to an upper case for titles
    # TODO: outline districts in black
    # TODO: if that still doesn't look good, turn off requirements for cmap to go [0,1]

    print("dem perc level", parties[0][2])
    plot_map(gdf, parties[0][2], 'Blues',
         f'Democratic Voters Percentage by {level}',
         f'figs/dem_perc_{level}.png')

    plot_map(gdf, parties[1][2], 'Reds',
         f'Republican Voters Percentage by {level}',
         f'figs/rep_perc_{level}.png')

    plot_map(gdf, parties[2][2], 'viridis',
         f'Nevada Independent Voters Percentage by {level}',
         f'figs/np_perc_{level}.png')

    # Reloading graph
    print("Loading updated graph...")
    graph = Graph.from_geodataframe(gdf)
    print("Updated graph loaded.")
    print(f"Information at nodes: {graph.nodes()[0].keys()}")
    print(f"Is the dual graph still connected? {nx.is_connected(graph)}")

    return partition, gdf, graph

congress_partition, gdf, graph = analyze_districts(gdf, "congress", graph, cut_edges)
assembly_partition, gdf, graph = analyze_districts(gdf, "assembly", graph, cut_edges)
senate_partition, gdf, graph = analyze_districts(gdf, "senate", graph, cut_edges)
commission_partition, gdf, graph = analyze_districts(gdf, "commission", graph, cut_edges)

print("Enacted maps data manipulation complete.\n")
print("gdf:\n", gdf.columns)


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
    if flag == "enacted": # Skip calculating Hispanic and Democratic ensembles for random start
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
    
        if flag == "enacted": # Run the full ensemble for the enacted plan
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
    breaaak
    
    # Histogram of number of cutedges in 2018 voting precincts
    plt.figure()
    plt.hist(cutedge_ensemble, edgecolor='black', color='purple')
    plt.xlabel("Cutedges", fontsize=12)
    plt.ylabel("Ensembles", fontsize=12)
    plt.suptitle("Cutedges in 2018 Voting Precincts",
              fontsize=14)
    plt.title(f"from {flag} start")
    plt.xlim(400, 850)  # Set x and y ranges so enacted-start and random-start ensembles
    plt.ylim(0, 3000)   # Are one-to-one comparisons
    plt.axvline(x=cutedges_enacted, color='orange', linestyle='--', linewidth=2,
                label=f"Enacted plan's cutedges = {cutedges_enacted}")
    plt.legend()
    plt.savefig(f"figs/histogram-cutedges-from-{flag}.png")
    print(f"Saved figs/histogram-cutedges-from-{flag}.png")
    if flag == "random":
        print("Random initial plan complete, terminating visualization early.")
        return
    
    # Histogram of number of Hispanic-30%+ districts from 2010 Census numbers
    plt.figure()
    plt.hist(h30_ensemble)
    plt.savefig("figs/histogram-hispanic.png")
    print("Saved figs/histogram-hispanic.png")
    
    # Specify boundaries between bins to make plot look a bit nicer
    plt.figure()
    plt.hist(h30_ensemble, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', color='orange')
    plt.xticks([0, 1, 2])
    plt.xlabel("Districts", fontsize=12)
    plt.ylabel("Ensembles", fontsize=12)
    plt.axvline(x=hisp_30_enacted, color='blue', linestyle='--', linewidth=2,
                label=f"Enacted plan's Hispanic population = {hisp_30_enacted}")
    plt.legend()
    plt.title("Districts with >30% Hispanic Population in 2010 Census",
              fontsize=14)
    plt.savefig("figs/histogram-hispanic-clean.png")
    print("Saved figs/histogram-hispanic-clean.png. You should double check the bins.")
    
    # Histogram of number of Democratic districts in US House race
    plt.figure()
    plt.hist(d_ensemble)
    plt.savefig("figs/histogram-democrats.png")
    print("Saved figs/histogram-democrats.png")
    
    # Specify boundaries between bins to make plot look a bit nicer
    plt.figure()
    plt.hist(d_ensemble, bins=[2.5, 3.5, 4.5, 5.5, 6.5], edgecolor='black', color='blue')
    plt.xticks([3, 4, 5, 6])
    plt.xlabel("Districts", fontsize=12)
    plt.ylabel("Ensembles", fontsize=12)
    plt.axvline(x=d_votes_enacted, color='orange', linestyle='--', linewidth=2,
                label=f"Enacted plan's Democratic districts = {d_votes_enacted}")
    plt.legend()
    plt.title("Democratic Districts in the 2018 Midterm Elections", fontsize=14)
    plt.savefig("figs/histogram-democrats-clean.png")
    print("Saved figs/histogram-democrats-clean.png. You should double check the bins.\n")

    # Make bopxlot
    a = np.array(hpop)
    district_stats_sorted = district_stats.sort_values('hisp_perc_cd')
    sorted_hpop = district_stats_sorted['hisp_perc_cd'].values
    plt.figure()
    plt.boxplot(a)
    plt.scatter(x = range(1, NUM_DIST + 1), y=sorted_hpop, color="red")
    plt.savefig("figs/boxplot-hispanic.png")

    plt.figure()
    plt.boxplot(a, patch_artist=True, 
            boxprops=dict(facecolor='orange', color='black'),
            medianprops=dict(color='blue', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            zorder=1)
    plt.scatter(x=range(1, NUM_DIST + 1), y=sorted_hpop, color="red", label="Enacted plan",
                zorder=2)
    plt.axhline(y=0.3, color='blue', linestyle='--', linewidth=2, label="30% threshold")
    plt.xticks(range(1, NUM_DIST + 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Districts", fontsize=12)
    plt.ylabel("Hispanic Percentage", fontsize=12)
    plt.title("Hispanic Population Distribution by District", fontsize=14)
    plt.legend()
    plt.savefig("figs/boxplot-hispanic-styled.png")

run_random_walk()
run_random_walk(enacted = False)

PATH_21_dir = "data/2021_Approved_Congressional_Plan_with_Final_Adjustments"
PATH_21 = PATH_21_dir + "/2021_Approved_Congressional_Plan_w_Final_Adjustments.shp"
print(f"\n2021 shapefile path: {PATH_21}\n")

# Dual graph from shapefile
print("Loading 2021 graph...")
graph_21 = Graph.from_file(PATH_21)
print("2021 graph loaded.\n")

print(f"Is the 2021 dual graph connected? {nx.is_connected(graph_21)}")
print(f"Is the 2021 dual graph planar? {nx.is_planar(graph_21)}")
print(f"Number of Nodes: {len(graph_21.nodes())}")
print(f"Number of Edges: {len(graph_21.edges())}")

print(f"Graph columns: {graph_21.nodes()[0].keys()}\n")
# dict_keys(['boundary_node', 'area', 'OBJECTID', 'District', 'Shape_Leng', 'Shape_Le_1',
# 'Shape_Area', 'geometry'])

# Geodataframe from shapefile
print("Loading 2021 Geodataframe...")
gdf_21 = gpd.read_file(PATH_21)
print("2021 geodataframe loaded.\n")
print(f"2021 shapefile columns: {gdf_21.columns}\n")

# Plot the new plan
plt.figure()
gdf_21.plot(cmap='tab10')
plt.title('Congressional Districts Today', fontsize=14)
plt.axis('off')
plt.savefig("figs/2021-congressional-districts.png")
print("Saved figs/2021-congressional-districts.png")
