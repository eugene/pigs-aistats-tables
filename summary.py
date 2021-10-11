from collections import defaultdict
from test_mean_equal import test_mean_difference
import pandas as pd

def process(df, higher_is_better=True, n=5):
    df = df.copy()
    
    # remove columns with nans
    df.dropna(axis=1, how='any', inplace=True)

    wins, draws, losses = defaultdict(int), defaultdict(int), defaultdict(int)

    for dataset in df.index.get_level_values(0):
        row = df.loc[dataset]
        methods = row.columns.unique(level=0).tolist()

        for method in methods:
            # Ensure zero's are there too.
            wins[method]   = wins[method] 
            draws[method]  = draws[method]
            losses[method] = losses[method]

            # Make sure we are not comparing to ourselfs. 
            compare_to_list = methods.copy()
            compare_to_list.remove(method)
            
            for compare_to in compare_to_list:
                dist_1 = {
                    "mean": row[method]['mean'].item(),
                    "std":  row[method]['std'].item(),
                    "n":    n,
                }

                dist_2 = {
                    "mean": row[compare_to]['mean'].item(),
                    "std":  row[compare_to]['std'].item(),
                    "n":    n,
                }

                if higher_is_better:
                    better_mean = dist_1['mean'] > dist_2['mean']
                else:
                    better_mean = dist_1['mean'] < dist_2['mean']

                stats_difrt = test_mean_difference(dist_1, dist_2)

                if better_mean         and stats_difrt:       wins[method] += 1
                elif better_mean       and (not stats_difrt): draws[method] += 1
                elif (not better_mean) and stats_difrt:       losses[method] +=1
                    
    return (wins, draws, losses)


df = pd.read_csv('ncp_sggm_uci_benchmarks.csv', index_col=[0,1], header=[0,1])

# Drop the `f_gaussian` method
df.drop('f_gaussian_noise', axis=1, level=0, inplace=True)

# Not shifted 
df = df.iloc[~df.index.get_level_values(0).str.endswith('_shifted')]

comparators = {
    'test_elbo↑':                    True,
    'test_expected_log_likelihood↑': True,
    'test_mean_fit_rmse↓':           False,
    'test_variance_fit_rmse↓':       False,
    'test_sample_fit_rmse↓':         False,
    'noise_mean_uncertainty↑':       True,
    'noise_kl_divergence↓':          False
}

class Data:
    def __init__(self):
        self.wins = self.draws = self.losses = 0

    def __repr__(self):
        return f"({self.wins}, {self.draws}, {self.losses})"

matrix = pd.DataFrame(0, index=comparators.keys(), columns=df.columns.unique(level=0).tolist())
matrix = matrix.astype(object)

for c in matrix:                   # Sorry. 
    for r, _ in matrix[c].items(): # Ain't got time 
        matrix.loc[r, c] = Data()  # for Pandas documentation.
        
for metric, higher_is_better in comparators.items():
    df_subset = df.loc[pd.IndexSlice[:, metric], :]
    wins, draws, losses = process(df_subset, higher_is_better = higher_is_better)

    for method, number in wins.items():
        matrix.loc[metric, method].wins += number

    for method, number in draws.items():
        matrix.loc[metric, method].draws += number

    for method, number in losses.items():
        matrix.loc[metric, method].losses += number
    
print(matrix)