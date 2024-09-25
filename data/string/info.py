import requests
import gzip
import pandas as pd

# URLs for the files
protein_info_url = 'https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz'
protein_links_url = 'https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz'

# Function to download and extract gz files
def download_and_extract_gz(url, output_filename):
    response = requests.get(url)
    with open(output_filename, 'wb') as f:
        f.write(response.content)
    with gzip.open(output_filename, 'rt') as f:
        return pd.read_csv(f, sep='\t')

# Download and read the protein info file
protein_info_df = download_and_extract_gz(protein_info_url, '9606.protein.info.v12.0.txt.gz')

# Download and read the protein links file
protein_links_df = download_and_extract_gz(protein_links_url, '9606.protein.links.v12.0.txt.gz')


# Show the first few rows of each dataframe
print("Protein Info DataFrame:")
print(protein_info_df.head())
print(protein_info_df.shape)

print("\nProtein Links DataFrame:")
print(protein_links_df.head())
print(protein_links_df.shape)
