from collections import defaultdict
from test_mean_equal import test_mean_difference
import sys
import pandas as pd

def process_one_win(df, higher_is_better=True, n=5):
    df = df.copy()
    wins, draws, losses = defaultdict(int), defaultdict(int), defaultdict(int)

    for i, dataset in enumerate(df.index.get_level_values(0)):
        row = df.loc[dataset]
        means_idx = row.columns.get_level_values(1).str.startswith('mean')
        row_means = row.iloc[:, means_idx].iloc[0]

        if higher_is_better:
            winner_method = row_means.idxmax()[0]
        else:
            winner_method = row_means.idxmin()[0]

        wins[winner_method] += 1
        
        # now we calculate the draws to other methods
        # for the same dataset
        looser_methods = row.columns.unique(level=0).tolist()
        looser_methods.remove(winner_method)

        for looser_method in looser_methods:
            dist_1 = {
                "mean": row[winner_method]['mean'].item(),
                "std":  row[winner_method]['std'].item(),
                "n":    n,
            }

            dist_2 = {
                "mean": row[looser_method]['mean'].item(),
                "std":  row[looser_method]['std'].item(),
                "n":    n,
            }
            
            if pd.isna(dist_1["mean"]) or pd.isna(dist_2["mean"]):
                continue

            stats_difrt = test_mean_difference(dist_1, dist_2)
            if not stats_difrt: draws[winner_method] += 1
    return (wins, draws, losses)

def process_accumulative(df, higher_is_better=True, n=5):
    df = df.copy()
    
    # remove columns with nans
    # df = df.dropna(axis=0, how='any')
    
    wins, draws, losses = defaultdict(int), defaultdict(int), defaultdict(int)

    for dataset in df.index.get_level_values(0):
        row = df.loc[dataset]
        methods = row.columns.unique(level=0).tolist()

        for method in methods:
            wins[method]   = wins[method]   # Ensures 
            draws[method]  = draws[method]  # zero's are 
            losses[method] = losses[method] # there too.

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
                
                if pd.isna(dist_1["mean"]) or pd.isna(dist_2["mean"]):
                    continue

                if higher_is_better:
                    better_mean = dist_1['mean'] > dist_2['mean']
                else:
                    better_mean = dist_1['mean'] < dist_2['mean']

                stats_difrt = test_mean_difference(dist_1, dist_2)

                if better_mean         and stats_difrt:       wins[method]   += 1
                elif better_mean       and (not stats_difrt): draws[method]  += 1
                elif (not better_mean) and stats_difrt:       losses[method] += 1
                # else:
                #     losses[method] += 1
                    
    return (wins, draws, losses)

class Data:
    def __init__(self):
        self.wins = self.draws = self.losses = 0
        self.winner = False

    def __repr__(self):
        return f"({self.wins}, {self.draws}, {self.losses})"

    def latex_tuple(self):
        return self.wins, self.draws, self.winner

    def latex_repr(self):
        return f"{self.wins} / {self.draws}"

def generate_latex(matrix, filename='output/0-table-summary.tex'):
    columns_full = {
        'f_kde':           r'd-VV',
        'f_standard_elbo': r'VV',
        'f_vap':           r'VV\\ (no prior)',
        'f_mvn':           r'Mean\\ variance\\ network',
        'f_ens':           r'Deep\\ ensembles',
        'f_mcd':           r'Monte\\ Carlo\\ dropout',
        'john':            r'Skafte\\ et al',
        'BBB+NCP':         r'Noise\\ contrastive\\ priors',
        'BBB':             r'Bayes\\ by\\ backprop',
        'Det':             r'Bifucarted\\ mean vari-\\ance network'
    }

    index_full = {
        'test_elbo↑':                    r'$\mathcal{L}$',
        'test_expected_log_likelihood↑': r'$\log p(y|x)$',
        'test_mean_fit_rmse↓':           r'$\text{RMSE}[y, \mu(x)]$',
        'test_variance_fit_rmse↓':       r'$\text{RMSE}[\mathrm{Var}[y|x], (y-\mu(x))^2]$', #[y|x],\left(y-\mu(x)\right)^2\right]$', 
        'test_sample_fit_rmse↓':         r'$\text{RMSE}[y,\tilde{y}]$',
        'noise_mean_uncertainty↑':       r'$\mathbb{E}[\sigma]$',
        'noise_kl_divergence↓':          r'$\mathbb{E}[\text{KL}]$'
    }

    matrix_ = matrix.rename(columns=columns_full, index=index_full)

    output = []
    # output.append(r'\definecolor{gray}{HTML}{aaaaaa}')
    output.append(r'\scalebox{0.85}{')
    output.append(r'\begin{tabular}{' + 'l' + 'r@{ / }lr@{ / }lr@{ / }lr@{ / }lr@{ / }l|r@{ / }lr@{ / }lr@{ / }lr@{ / }l' + r'}')
    output.append(r'\toprule')
    mcols = list(map(lambda e: "\makecell[l]{" + e + "}", matrix_.columns))
    # mcols = list(map(lambda e: "\multicolumn{2}{r}{" + e + "}", mcols))
    mcols_ = []
    for i, col in enumerate(mcols):
        if i == 4:
            mcols_.append("\multicolumn{2}{r|}{" + col + "}")
        else:
            mcols_.append("\multicolumn{2}{r}{" + col + "}")
    output.append(r'\makecell{UCI benchmarks\\ shifts included} & ' + r' & '.join(mcols_) + r"\\")
    output.append(r'\midrule')

    for i, row in matrix_.iterrows():
        row_ = [i]
        for k, v in row.iteritems():
            wins, draws, winner = v.latex_tuple()
            if wins != -1:
                wins_str = str(wins) if not winner else r'\textbf{' + str(wins) + r"}"
                row_.append(r"{" + wins_str + r"}&{\color{gray}{" + str(draws) + r"}}")
            else:
                row_.append(r"{\color{gray}{n}}&{\color{gray}{a}}")

        output.append( ' & '.join(row_) + r" \\")
        # break

    output.append(r'\bottomrule')
    output.append(r'\end{tabular}')
    output.append(r'}')

    with open(filename, "w") as text_file:
        text_file.write("\n".join(output))

    return matrix_

df = pd.read_csv('ncp_sggm_uci_benchmarks.csv', index_col=[0,1], header=[0,1])

# Drop the `f_gaussian` and `Det` method
df.drop('f_gaussian_noise', axis=1, level=0, inplace=True)
df.drop('Det', axis=1, level=0, inplace=True)

# Re-arrange
cols = ['f_kde', 'f_standard_elbo', 'f_vap', 'f_mvn', 'john', 'f_ens' , 'f_mcd', 'BBB+NCP', 'BBB']
new_cols = df.columns.reindex(cols, level=0)
df = df.reindex(columns=new_cols[0]) #new_cols is a single item tuple

# sys.exit()

# Not shifted or shifted
# df = df.iloc[~df.index.get_level_values(0).str.endswith('_shifted')]
# df = df.iloc[df.index.get_level_values(0).str.endswith('_shifted')]

comparators = {
    'test_elbo↑':                    True,
    'test_expected_log_likelihood↑': True,
    'test_mean_fit_rmse↓':           False,
    'test_variance_fit_rmse↓':       False,
    'test_sample_fit_rmse↓':         False,
    # 'noise_mean_uncertainty↑':       True,
    'noise_kl_divergence↓':          False
}

if True or not 'matrix' in locals():
    # allows to cache the calculation when running
    # %run -i summary.py 
    # in ipython
    matrix = pd.DataFrame(0, index=comparators.keys(), columns=df.columns.unique(level=0).tolist())
    matrix = matrix.astype(object)

    for c in matrix:                   # Sorry. Ain't got time 
        for r, _ in matrix[c].items(): # for Pandas documentation.
            matrix.loc[r, c] = Data()  # The one liner was acting up.

    for metric, higher_is_better in comparators.items():
        df_subset = df.loc[pd.IndexSlice[:, metric], :]

        # wins, draws, losses = process_accumulative(df_subset, higher_is_better = higher_is_better)
        wins, draws, losses = process_one_win(df_subset, higher_is_better = higher_is_better)

        best_val, best_method = 0, None

        for method, number in wins.items():
            matrix.loc[metric, method].wins += number
            if number > best_val:
                best_val = number
                best_method = method

        matrix.loc[metric, best_method].winner = True

        for method, number in draws.items():
            matrix.loc[metric, method].draws += number

        for method, number in losses.items():
            matrix.loc[metric, method].losses += number

    # Where do we have NA's in original dataframe?
    na_boolean_df = df.isna().groupby(level=[1]).sum() == len(df.index.get_level_values(0).unique())
    na_boolean_df = na_boolean_df.groupby(axis=1, level=[0]).sum() > 1
    
    for c in na_boolean_df:                   
        for r, _ in na_boolean_df[c].items():
            if (True == na_boolean_df.loc[r,c]) and (r in matrix.index):
                matrix.loc[r, c].wins = -1

generate_latex(matrix)