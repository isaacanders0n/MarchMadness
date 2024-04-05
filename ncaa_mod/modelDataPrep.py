import pandas as pd
import json
base_path = ''

file_paths = [base_path + 'cbb13.csv', base_path + 'cbb14.csv', base_path + 'cbb15.csv', 
              base_path + 'cbb16.csv', base_path + 'cbb17.csv', base_path + 'cbb18.csv', 
              base_path + 'cbb19.csv', base_path + 'cbb20.csv', base_path + 'cbb21.csv', 
              base_path + 'cbb22.csv', base_path + 'cbb23.csv']

unique_teams_per_file = {}

files_corrected = {
    "2013": "cbb13.csv",
    "2014": "cbb14.csv",
    "2015": "cbb15.csv",
    "2016": "cbb16.csv",
    "2017": "cbb17.csv",
    "2018": "cbb18.csv",
    "2019": "cbb19.csv",
    "2020": "cbb20.csv",
    "2021": "cbb21.csv",
    "2022": "cbb22.csv",
    "2023": "cbb23.csv",
}

for file_path in file_paths:
    df = pd.read_csv(file_path)
    unique_teams_per_file[file_path] = set(df['TEAM'].unique())
universal_teams = set.union(*unique_teams_per_file.values())

missing_teams_per_file = {}

for file_path, teams in unique_teams_per_file.items():
    missing_teams_per_file[file_path] = universal_teams - teams
data_for_csv = []
for file_path, missing_teams in missing_teams_per_file.items():
    year = file_path.split('/')[-1].replace('cbb', '').replace('.csv', '')
    
    for team in missing_teams:
        data_for_csv.append([year, team])

missing_teams_df = pd.DataFrame(data_for_csv, columns=['Year', 'Missing Team'])
base_path = ''
missing_teams_df.to_csv(base_path + 'missing_teams.csv', index=False)

df = pd.read_csv(base_path + 'missing_teams.csv')
df['Year'] = '20' + df['Year'].astype(str)

reshaped_df = df.pivot_table(index=[df.index], columns=['Year'], values='Missing Team', aggfunc='first').reset_index(drop=True)
reshaped_df = reshaped_df.dropna(axis=1, how='all')

reshaped_df_corrected = pd.pivot_table(df, values='Missing Team', index=[df.groupby('Year').cumcount()], columns=['Year'], aggfunc='first')
reshaped_df_filled = reshaped_df_corrected.fillna('')

year_team_dict_corrected = {year: reshaped_df_filled[year].dropna().tolist() for year in reshaped_df_filled.columns}
year_team_dict_cleaned = {year: [team for team in teams if team] for year, teams in year_team_dict_corrected.items()}

save_path = base_path + 'missing_teams.json'

with open(save_path, 'w') as f:
    json.dump(year_team_dict_cleaned, f, indent=4)

with open(save_path, 'r') as f:
    missing_teams_data = json.load(f)

def extract_unique_teams(json_data):
    unique_teams = set()
    for year_teams in json_data.values():
        for team in year_teams:
            unique_teams.add(team)
    return list(unique_teams)

unique_teams_to_remove = extract_unique_teams(missing_teams_data)
processed_csv_path = 'cleaned_csvs/'
suffix = '.csv'
prefix = 'cleaned-cbb'
output_paths = [processed_csv_path + prefix + year + suffix for year in missing_teams_data.keys()]

def preprocess_and_verify_v2(input_paths, output_paths, teams_to_remove):
    for input_path, output_path in zip(input_paths.values(), output_paths):
        df = pd.read_csv(input_path)
        df.drop(columns=['SEED'], inplace=True, errors='ignore')
        if 'POSTSEASON' in df.columns:
            df['POSTSEASON'].fillna('DNC', inplace=True)
        if 'RK' in df.columns:
            df.drop(columns=['RK'], inplace=True, errors='ignore')
        if 'EFGD_D' in df.columns:
            df['EFG_D'] = df['EFGD_D']
            df.drop(columns=['EFGD_D'], inplace=True, errors='ignore')
        df = df[~df['TEAM'].isin(teams_to_remove)]
        df.to_csv(output_path, index=False)
        verify_df = pd.read_csv(output_path)
        assert not verify_df['TEAM'].isin(teams_to_remove).any(), "Verification failed: Teams not removed correctly."
        assert 'SEED' not in verify_df.columns, "Verification failed: SEED column not dropped."
        if 'POSTSEASON' in df.columns:
            assert verify_df['POSTSEASON'].isnull().sum() == 0, "Verification failed: POSTSEASON NaN values not replaced."
        if 'RK' in df.columns:
            assert 'RK' not in verify_df.columns, "Verification failed: RK column not dropped."
        if 'EFGD_D' in df.columns:
            assert 'EFGD_D' not in verify_df.columns, "Verification failed: EFGD_D column not dropped."
        assert 'EFG_D' in verify_df.columns, "Verification failed: EFG_D column not present."

preprocess_and_verify_v2(files_corrected, output_paths, unique_teams_to_remove)
base_cleaned = 'cleaned_csvs/cleaned-cbb'
cleaned_file_column_amounts = [pd.read_csv(base_cleaned + year + '.csv').shape[1] for year in files_corrected.keys()]