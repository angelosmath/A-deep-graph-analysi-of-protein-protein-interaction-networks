import os
import pandas as pd
import gzip
from alive_progress import alive_bar

class UCSC_pplink:
    def __init__(self):
        self.url = 'http://hgdownload.soe.ucsc.edu/goldenPath/hgFixed/database/ggLink.txt.gz'
        self.output_filename = 'ggLink.txt.gz'

    def download_pplink(self):
        if not os.path.exists(self.output_filename):
            try:
                os.system(f"wget {self.url} -O {self.output_filename}")
            except Exception as e:
                print(e)

    def to_dataframe(self):
        lines = []
        with gzip.open(self.output_filename, 'rb') as f:
            for line in f:
                lines.append(line.decode('utf-8'))

        df_rows = []
        with alive_bar(round(len(lines)), title='lines processed') as bar:
            for line in lines:
                bar()
                gene1, gene2, linkTypes, pairCount, oppCount, docCount, dbList, minResCount, snippet, context = line.split("\t")
                df_rows.append([gene1, gene2, linkTypes, pairCount, oppCount, docCount, dbList, minResCount, snippet, context])

        df = pd.DataFrame(df_rows, columns=["gene1", "gene2", "linkTypes", "pairCount", "oppCount", "docCount", "dbList", "minResCount", "snippet", "context"])

        # Clean data from single numerical values ('2')
        df = df.loc[~df['gene1'].str.isdigit() & ~df['gene2'].str.isdigit()]

        # Keep only ppi relations
        df = df[df['linkTypes'] == 'ppi']

        #df = df.iloc[:, :3]
        df.to_csv('gg_ppi.csv', index=False)

def main():
    gglink = UCSC_pplink()
    gglink.download_pplink()
    gglink.to_dataframe()

if __name__ == "__main__":
    main()