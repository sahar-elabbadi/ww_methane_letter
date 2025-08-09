import pandas as pd
import pathlib
# 1) Load the CSV file
file_path = pathlib.Path("01_raw_data", "Wechselberger2025_SI-data.csv")  # Update path if needed
df = pd.read_csv(file_path)

# 2) Select rows where data_source_citation matches
filtered_df = df[df['data_source_citation'] == "Fredenslund et al., 2023"]
