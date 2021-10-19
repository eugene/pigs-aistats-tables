from collections import defaultdict
import sys
import pandas as pd

df = pd.read_csv('ncp_sggm_uci_benchmarks.csv', index_col=[0,1], header=[0,1])

# Drop the `f_gaussian` and `Det` method
df.drop('f_gaussian_noise', axis=1, level=0, inplace=True)
df.drop('Det', axis=1, level=0, inplace=True)

# Re-arrange
cols = ['f_kde', 'f_standard_elbo', 'f_vap', 'f_mvn', 'john', 'f_ens' , 'f_mcd', 'BBB+NCP', 'BBB']
new_cols = df.columns.reindex(cols, level=0)
df = df.reindex(columns=new_cols[0]) #new_cols is a single item tuple

columns_full = {
    'f_kde':           r'd-VV',
    'f_standard_elbo': r'VV',
    'f_vap':           r'VV\\ (no prior)',
    'f_mvn':           r'Mean\\ variance\\ network',
    'john':            r'Skafte\\ et al',
    'f_ens':           r'Deep\\ ensembles',
    'f_mcd':           r'Monte\\ Carlo\\ dropout',
    'BBB+NCP':         r'Noise\\ contrastive\\ priors',
    'BBB':             r'Bayes\\ by\\ backprop',
    # 'Det':           r'Bifucarted\\ mean vari-\\ance network'
}

# test_elbo↑
# test_expected_log_likelihood↑
# test_mean_fit_rmse↓
# test_variance_fit_rmse↓
# test_sample_fit_rmse↓
# noise_kl_divergence↓

metric_name = 'test_sample_fit_rmse↓'
                           
metric = df.loc[pd.IndexSlice[:, metric_name], :]
subset = metric.iloc[~metric.index.get_level_values(0).str.endswith('_shifted')]
subset2 = metric.iloc[metric.index.get_level_values(0).str.endswith('_shifted')]

output = []
output.append(r'\resizebox{\columnwidth}{!}{')
# output.append(r'\setcellgapes{0pt}\makegapedcells')
align_string = "r@{ $\pm$ }l"*4
align_string += "r@{ $\pm$ }l|"
align_string += "r@{ $\pm$ }l"*4
output.append(r'\begin{tabular}{' + '@{\hskip0pt}ll' + align_string + r'}')
output.append(r'\toprule')
mcols = list(map(lambda e: "\makecell[l]{" + e + "}", columns_full.values()))
mcols_ = []
for i, col in enumerate(mcols):
    if i == 4:
        mcols_.append("\multicolumn{2}{r|}{" + col + "}")
    else:
        mcols_.append("\multicolumn{2}{r}{" + col + "}")
        
output.append(r'\multicolumn{2}{l}{\makecell{UCI benchmarks}} & ' + r' & '.join(mcols_) + r"\\")
output.append(r'\midrule')

for ii, (index, row) in enumerate(subset.iterrows()):
    line, cells = [], []
    prepend = r'\multirow{12}{*}{\rotatebox[origin=c]{90}{Not shifted}}' if ii == 0 else '~'
    line.append(prepend + r' & \texttt{' + index[0].replace('_', r'\_') + r'}')
    for enum, (i_, r) in enumerate(row.items()):
        cells.append(str(round(r, 2)))
    line.append('&'.join(cells))
    output.append(r' & '.join(line) + r' \\')

output.append(r'\midrule')

for ii, (index, row) in enumerate(subset2.iterrows()):
    line, cells = [], []
    prepend = r'\multirow{12}{*}{\rotatebox[origin=c]{90}{Shifted}}' if ii == 0 else '~'
    line.append(prepend + r' & \texttt{' + index[0].replace('_', r'\_').replace(r'\_shifted', '') + r'}')
    for enum, (i_, r) in enumerate(row.items()):
        cells.append(str(round(r, 2)))
    line.append('&'.join(cells))
    output.append(r' & '.join(line) + r' \\')


output.append(r'\bottomrule')
output.append(r'\end{tabular}')
output.append(r'}')


with open(f"output/full-table-{metric_name}.tex", "w") as text_file:
    text_file.write("\n".join(output))