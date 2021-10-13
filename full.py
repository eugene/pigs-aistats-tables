from numpy import longfloat
import pandas as pd

df = pd.read_csv('ncp_sggm_uci_benchmarks.csv')
new_col_names = { 
    df.columns[0]: "dataset",
    df.columns[1]: "metric",
}
df.rename(columns=new_col_names, inplace = True)
df.loc[0]['dataset'] = df.loc[0]['metric'] = ''

def get_df(metric = 'test_elbo↑', shifted=False):
    if not shifted:
        not_shifted = ~df['dataset'].str.endswith('_shifted')
    else:
        not_shifted = df['dataset'].str.endswith('_shifted')

    metric      = df['metric'] == metric

    subset = df[metric & not_shifted]

    resulting_series = {}
    columns_of_interest = subset.columns[2:][~subset.columns[2:].str.endswith('.1')]

    for col in columns_of_interest:
        values_series = subset[col].astype(float).round(2).astype(str)
        std_series = subset[col + '.1'].astype(float).round(2).astype(str)
        series = values_series + '±' + std_series
        series.replace('nan±nan', 'na', inplace=True)
        resulting_series[col] = series #.append(series)

    resulting_df = pd.DataFrame(resulting_series)
    resulting_df.index = subset['dataset']

    order = ['f_kde', 'f_standard_elbo', 'f_gaussian_noise', 'f_vap', 'f_mvn', 'f_ens', 'f_mcd', 'john', 'BBB+NCP', 'BBB', 'Det']
    resulting_df = resulting_df[order]
    resulting_df.index.name = ''

    new_col_names = { 
        resulting_df.columns[0]: "d-VV",
        resulting_df.columns[1]: "VV (no PIG)",
        resulting_df.columns[2]: "VV (Gaussian Noise)",
        resulting_df.columns[3]: "VV (no prior)",
    }
    resulting_df.rename(columns=new_col_names, inplace = True)
    return resulting_df

def write_to_file(path, df, caption):
    kwargs = {
        'longtable': True,
        'position': 'l',
        'bold_rows': True,
        'index_names': False,
    }
    with open(path, "w") as text_file:
        text_file.write(df[df.columns[:5]].to_latex(**kwargs, caption=caption))
        text_file.write('\\addtocounter{table}{-1}\n')
        text_file.write(df[df.columns[5:]].to_latex(**kwargs))

df_uci_L_not_shifted = get_df(metric = 'test_elbo↑', shifted=False)
write_to_file("output/1-table_elbo_not_shifted.tex", df_uci_L_not_shifted, "UCI benchmarks - $\mathcal{L}$")

df_uci_Ell_not_shifted = get_df(metric = 'test_expected_log_likelihood↑', shifted=False)
write_to_file("output/2-table_ELL.tex", df_uci_Ell_not_shifted, "UCI benchmarks - $\log p(y|x)$")

df_uci_L_not_shifted = get_df(metric = 'test_mean_fit_rmse↓', shifted=False)
write_to_file("output/3-table_RMSE.tex", df_uci_L_not_shifted, "UCI benchmarks - RMSE $[y, \mu(x)]$")