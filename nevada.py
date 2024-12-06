from functools import partial
import warnings
import argparse
import random
import networkx as nx
import numpy as np
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

# Parse the arguments
# args = parser.parse_args()

# Use the total_steps argument in your code
# total_steps = args.total_steps

# Set global matplotlib dpi for high quality plots
# plt.rcParams['savefig.dpi'] = 300

# Set path to shapefile
PATH = "data/precinct_p_fixed.shp"
print(f"Shapefile path: {PATH}\n")

# Geodataframe from shapefile
print("Loading Geodataframe...")
gdf = gpd.read_file(PATH)
print("Geodataframe loaded.\n")

# Check total rows and general info
gdf.info(verbose=True)

# Dual graph from shapefile
print("Loading Graph...")
graph = Graph.from_geodataframe(gdf)
print("Graph loaded.\n")

print(f"Graph columns: {graph.nodes()[0].keys()}\n")

# Check if 'PREC' is unique
is_unique = gdf['PREC'].is_unique
print("Is 'PREC' unique?:", is_unique)

# Count unique values in 'PREC'
unique_count = gdf['PREC'].nunique()
print("Number of unique 'PREC' values:", unique_count)

# Check for missing values in 'PREC'
missing_prec = gdf['PREC'].isna().sum()
print("Number of missing 'PREC' values:", missing_prec)

# Get the most frequent values in 'PREC'
print("Most frequent 'PREC' values:\n", gdf['PREC'].value_counts().head())

# Get a summary of the GeoDataFrame
print(gdf.info())

# Check the geometry column
print("Geometry type counts:\n", gdf.geom_type.value_counts())

print("Geometry type:", gdf['geometry'].geom_type.unique())

# Plot precinct shapefile
plt.figure()
ax = gdf.plot(
    column=None,  # No specific column, just the geometry
    edgecolor='black',
    linewidth=0.2
)

# Suppress axes and add title
plt.axis('off')
plt.title('Clark County Precincts', fontsize=14)

# Save the figure
plt.savefig("figs/precincts.png")
print("\nSaved figs/precincts.png")

print(f"Shapefile columns: {gdf.columns}\n")
# Shapefile columns: Index(['PREC', 'WARD', 'COMMISSION', 'ASSEMBLY', 'SENATE', 'EDUCATION',
       # 'REGENT', 'SCHOOL', 'CONGRESS', 'TOWNSHIP', 'POLLING', 'GlobalID',
       # 'Shape_Leng', 'geometry'],
      # dtype='object')
breaaak

# Print populations
# hisp = sum(graph.nodes()[v]['HISP'] for v in graph.nodes())
# print(f"Hispanic Population: {hisp}")
# totpop = sum(graph.nodes()[v]['TOTPOP'] for v in graph.nodes())
# print(f"Total Population: {totpop}")
# print(f"Hispanic percentage: {hisp/totpop:.2%}")

# Add Hispanic percentage from the 2010 Census to the geodataframe
# gdf['hisp_perc'] = gdf['HISP']/gdf['TOTPOP']

# Plot the Hispanic percentage
# plt.figure()
# ax = gdf.plot()
# sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=0, vmax=1)) # Custom legend
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=-0.01, shrink=0.7)
# plt.axis('off') # To suppress axes in the map plot
# plt.title('Hispanic Percentage in 2010 Census', fontsize=14)
# plt.savefig("hispanic.png")
# print("\nSaved figs/hispanic.png")

# Add Democratic votes percentage in 2018 US House race to the geodataframe
gdf['dem_perc'] = gdf["USH18D"]/gdf[["USH18D", "USH18R"]].sum(axis=1)

# Plot the partisan distribution
plt.figure()
ax = gdf.plot(column = 'dem_perc', cmap = 'seismic_r', vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='seismic_r', norm=plt.Normalize(vmin=0, vmax=1)) # Custom legend
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=-0.01, shrink=0.7)
cbar.set_label('Percent of Votes', fontsize=12)
cbar.set_ticks([0, 1])  # Position tick labels at 0 (Republican) and 1 (Democratic)
cbar.set_ticklabels(['Republican', 'Democratic'])  # Set custom labels
plt.axis('off') # To suppress axes in the map plot
plt.title('Votes for US House in 2018 Midterms', fontsize=14)
plt.savefig("figs/party.png")
print("Saved figs/party.png\n")

# Redefine new graph object with the hisp_perc and dem_perc columns
# print("Loading updated graph...")
# graph = Graph.from_geodataframe(gdf)
# print("Updated graph loaded.")
# print(f"Information at nodes: {graph.nodes()[0].keys()}")

# Create a partition object from the enacted "CD116FP" districting plan
print("Turning enacted congressional map into a partition...")
enacted_plan = GeographicPartition(graph,
                                assignment= "CD116FP",
                                updaters = {
                                    "cutedges": cut_edges, 
                                    }
                                )

print("Running aggregation into congressional districts...")
# Aggregate data using groupby
district_stats = gdf.groupby('CD116FP').agg({
    'TOTPOP': 'sum',
    'HISP': 'sum',
    'USH18D': 'sum',
    'USH18R': 'sum'
}).reset_index()

# Calculate percentage of Hispanic population and Democratic votes
district_stats['hisp_perc_cd'] = district_stats['HISP'] / district_stats['TOTPOP']
district_stats['dem_perc_cd'] = district_stats['USH18D'] / (district_stats['USH18D'] + district_stats['USH18R'])

# Calculate the number of 30%+ Hispanic districts
hisp_30_enacted = (district_stats['hisp_perc_cd'] >= 0.3).sum()
print(f"Number of districts with 30% or more Hispanic population: {hisp_30_enacted}")

# Calculate the number of Democratic districts
d_votes_enacted = (district_stats['USH18D'] > district_stats['USH18R']).sum()
print(f"Number of districts Democrats won in 2018 House elections: {d_votes_enacted}")

# Calculate number of cutedges
cutedges_enacted = len(enacted_plan['cutedges'])
print(f"Number of cutedges: {cutedges_enacted}")

# Merge the aggregated data back into the GeoDataFrame
gdf = gdf.merge(district_stats[['CD116FP', 'hisp_perc_cd', 'dem_perc_cd']], on='CD116FP', how='left')

print("Enacted congressional map data manipulation complete.\n")

# Plot the Hispanic percentage by congressional district
plt.figure()
# Use a scale of 0 to 0.3 even though no values are at 0.3 to indicate falling below 30%
# This value will become important in the ensemble analysis
ax = gdf.plot(column='hisp_perc_cd', cmap='Oranges', vmin=0, vmax=0.3)
sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=0, vmax=0.3))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.01, shrink=0.7)
plt.axis('off') # To suppress axes in the map plot
plt.title('Hispanic Percentage in 2010 Census by Congressional District', fontsize=14)
plt.savefig("figs/hispanic_cd.png")
print("Saved figs/hispanic_cd.png")

# Plot the partisan vote makeup by congressional district
plt.figure()
ax = gdf.plot(column='dem_perc_cd', cmap='seismic_r', vmin=0, vmax=1)
plt.axis('off') # To suppress axes in the map plot
plt.title('Votes for US House in 2018 Midterms by Congressional District', fontsize=14)
sm = plt.cm.ScalarMappable(cmap='seismic_r', norm=plt.Normalize(vmin=0, vmax=1)) # Custom legend
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.01, shrink=0.7)
cbar.set_label('Percent of Votes', fontsize=12)
cbar.set_ticks([0, 1])  # Position tick labels at 0 (Republican) and 1 (Democratic)
cbar.set_ticklabels(['Republican', 'Democratic'])  # Set custom labels
plt.savefig("figs/party_cd.png")
print("Saved figs/party_cd.png")

# Plot congressional districts with no coloring
plt.figure()
gdf.plot(pd.Series([enacted_plan.assignment[i] for i in gdf.index]), cmap="tab10")
plt.title('Congressional Districts in 2018', fontsize=14)
plt.axis('off')
plt.savefig("figs/2012-congressional-districts.png")
print("Saved figs/2012-congressional-districts.png\n")

# Make an initial districting plan using recursive_tree_part
print("Setting up ensemble..")
NUM_DIST = 7 # for the seven Congressional districts Colorado had in 2018
ideal_pop = totpop/NUM_DIST
POP_TOLERANCE = 0.02

def run_random_walk(enacted = True):
    if enacted:
        flag = "enacted"
        initial_plan = "CD116FP"
    else:
        flag = "random"
        initial_plan = recursive_tree_part(graph,
                                           range(NUM_DIST),
                                           ideal_pop,
                                           'TOTPOP',
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
            "population": Tally("TOTPOP", alias = "population"), 
            "Hispanic population": Tally("HISP", alias = "Hispanic population"),
            "R votes": Tally("USH18R", alias = "R votes"),
            "D votes": Tally("USH18D", alias = "D votes")
        }
    )
    
    # Set up random walk
    rw_proposal = partial(recom,                   # How you choose a next districting plan
                          pop_col = "TOTPOP",      # What data describes population
                          pop_target = ideal_pop,  # Target/ideal population is for each district
                          epsilon = POP_TOLERANCE, # How far from ideal population you can deviate
                          node_repeats = 1         # Number of times to repeat bipartition.
                          )
    
    # Set up population constraint
    population_constraint = constraints.within_percent_of_ideal_population(
            initial_partition,
            POP_TOLERANCE,
            pop_key = "population"
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
        h30_ensemble = []
        d_ensemble = []
        hpop = []
    
    for part in our_random_walk:
        # Add cutedges to cutedges ensemble
        cutedge_ensemble.append(len(part["cutedges"]))
    
        if flag == "enacted": # Run the full ensemble for the enacted plan
            hisp_30 = 0
            d_votes = 0
            hpop_this_step = []

            for district in part.parts:
                # 30%+ Hispanic districts from US Census
                h_perc = part["Hispanic population"][district]/part["population"][district]
                if h_perc > 0.3:
                    hisp_30 += 1
                hpop_this_step.append(h_perc)

                # Districts with more D votes than R votes in 2018 US House race
                if part["D votes"][district] > part["R votes"][district]:
                    d_votes += 1

            h30_ensemble.append(hisp_30)
            hpop_this_step.sort()
            hpop.append(hpop_this_step)
            d_ensemble.append(d_votes)
    
    print("Random walk complete.\n")
    
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
