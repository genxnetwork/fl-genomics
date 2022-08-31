import pandas as pd

# read Neale Lab pan-ancestry UKB phenotype manifest
pheno_man = pd.read_csv('/mnt/genx-bio-share/TG/data/chip/external/nealelab/h2_manifest.tsv', sep='\t')
pheno_man = pheno_man.query("pheno_sex == 'both_sexes'")
pheno_man = pheno_man[pheno_man['trait_type'].isin(['continuous', 'biomarkers'])]

# leave only phenocodes with three pops
df = pheno_man[pheno_man['pop'].isin(['EUR', 'AFR', 'CSA'])]
three_rows = df.groupby('phenocode')['pop'].count() == 3
df = df[df['phenocode'].isin(three_rows[three_rows].index)]

# leave only phenocodes with all ancestries passing qc
all_anc_pass = df.groupby('phenocode')['qcflags.pass_all'].sum() == 3
df_pass = df[df['phenocode'].isin(all_anc_pass[all_anc_pass].index)]

# sort by min observed h2 across 3 ancestries
min_obs_h2 = df_pass.groupby('phenocode')['estimates.final.h2_observed'].min()
df_pass = df_pass.set_index(['phenocode', 'pop']).join(min_obs_h2.rename('min_obs_h2')).sort_values('min_obs_h2', ascending=False)

# add phenotype names
pheno_names = pd.read_csv('/mnt/genx-bio-share/TG/data/chip/external/nealelab/Pan-UK Biobank phenotype manifest - phenotype_manifest.csv')[['phenocode', 'description']].dropna().drop_duplicates().set_index('phenocode')['description']
pheno_names = pheno_names[~pheno_names.index.duplicated(keep=False)]
df_pass = pheno_names.to_frame().join(df_pass, how='right').reset_index()
df_pass_display = df_pass[['phenocode', 'pop', 'description', 'trait_type', 'min_obs_h2', 'estimates.final.h2_observed']]

df_shortlist = df_pass_display[df_pass_display['phenocode'].isin(['50', '30080', '30100', '23104', '23105', '30300', '3062', '30010', '30250', '30270', '30870', '30620'])]
pass