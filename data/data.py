import os
import gzip
import pandas as pd
from alive_progress import alive_bar
import requests
import zipfile
from io import BytesIO

class Data:
    """
    Class to handle downloading and processing project data
    """

    def __init__(self):
        self.string_url = 'https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz'
        self.ucsc_url = 'http://hgdownload.soe.ucsc.edu/goldenPath/hgFixed/database/ggLink.txt.gz'
        self.negatome_url = 'http://mips.helmholtz-muenchen.de/proj/ppi/negatome/'
        self.chorum_url = 'https://mips.helmholtz-muenchen.de/corum/download/releases/current/humanComplexes.txt.zip'

    def download(self, url):
        """
        Downloads a file from the specified URL if it does not already exist.
        """
        filename = url.split('/')[-1]
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                os.system(f"wget {url} -O {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists. Skipping download.")
        return filename

    def string(self):
        """
        Processes STRING data.
        """
        try:
            filename = self.download(self.string_url)
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                lines = [line.strip() for line in f]

            processed_lines = []
            with alive_bar(len(lines), title='Processing STRING data') as bar:
                for line in lines:
                    bar()
                    if '.' in line:
                        elements = line.split()
                        elements[0] = elements[0].split('.')[1]
                        elements[1] = elements[1].split('.')[1]
                    else:
                        elements = line.split()
                    processed_lines.append(elements)

            df = pd.DataFrame(processed_lines[1:], columns=processed_lines[0])
            df.to_csv('string_db_interactions.csv', index=False)
            print("STRING data processed successfully.")
        except Exception as e:
            print(f"Error processing STRING data: {e}")

    def ucsc(self):
        """
        Processes UCSC PPI data.
        """
        try:
            filename = self.download(self.ucsc_url)
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                lines = f.readlines()

            df_rows = []
            with alive_bar(len(lines), title='Processing UCSC data') as bar:
                for line in lines:
                    bar()
                    elements = line.strip().split("\t")
                    if len(elements) == 10:  # Ensure correct number of columns
                        df_rows.append(elements)

            df = pd.DataFrame(df_rows, columns=["gene1", "gene2", "linkTypes", "pairCount", "oppCount", 
                                                "docCount", "dbList", "minResCount", "snippet", "context"])

            df = df.loc[~df['gene1'].str.isdigit() & ~df['gene2'].str.isdigit()]
            df = df[df['linkTypes'] == 'ppi']
            df.to_csv('gg_ppi.csv', index=False)
            print("UCSC data processed successfully.")
        except Exception as e:
            print(f"Error processing UCSC data: {e}")

    def negatome(self):
        """
        Process Negatome v2.0 data.
        """
        try:
            filename = self.download(self.negatome_url)
        except Exception as e:
            print(f"Error processing CORUM data: {e}")

    def chorum(self):
        """
        Processes CORUM data from a ZIP file containing a GZ file, and saves it to a CSV file.
        """
        filename = self.download(self.chorum_url)

        # Check if the downloaded file is a valid ZIP file
        if not zipfile.is_zipfile(filename):
            print(f"Error: {filename} is not a valid ZIP file.")
            return

        try:
            with zipfile.ZipFile(filename, 'r') as z:
                # Extract the .txt file from the zip
                z.extractall()  # Extract to the current directory
                print(f"Extracted file: {z.namelist()}")

                # Check the extracted files
                extracted_files = z.namelist()
                if len(extracted_files) == 1:
                    extracted_file = extracted_files[0]
                else:
                    print("Unexpected number of files extracted.")
                    return
        except Exception as e:
            print(f"Error processing CORUM data: {e}")
            return
        
        # Process the extracted file (ensure it's the correct file)
        print(f"Processing extracted file: {extracted_file}")

        try:
            # Read the extracted .txt file
            df = pd.read_csv(extracted_file, sep='\t')

            print('DataFrame shape:', df.shape)

            # Exclude rows with NaN values in 'subunits(Gene name)' or 'GO ID'
            df = df.dropna(subset=['subunits(Gene name)', 'GO ID'])

            # Remove duplicates
            df = df.drop_duplicates(subset=['ComplexName'])

            # Split the gene names and GO terms into lists
            df['subunits_gene_list'] = df['subunits(Gene name)'].apply(lambda x: x.split(';'))
            df['GO_terms_list'] = df['GO ID'].apply(lambda x: x.split(';'))

            # Add a column to count the number of gene members in each complex
            df['num_genes'] = df['subunits_gene_list'].apply(len)

            # Save to CSV
            df.to_csv('human_complexes.csv', index=False)
            print("Data saved to 'human_complexes.csv'.")

        except Exception as e:
            print(f"Error processing extracted file: {e}")


if __name__ == "__main__":
    data = Data()
    data.string()
    data.ucsc()
    data.negatome()
    data.chorum()
