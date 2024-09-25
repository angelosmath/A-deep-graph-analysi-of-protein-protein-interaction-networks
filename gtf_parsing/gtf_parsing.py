import os
import gzip
import pandas as pd
from alive_progress import alive_bar

class GTF_parsing:
    
    def __init__(self):
        self.pplink_url = 'http://ftp.ensembl.org/pub/release-104/gtf/homo_sapiens/Homo_sapiens.GRCh38.104.gtf.gz'
        self.filename = 'Homo_sapiens.GRCh38.104.gtf.gz'

    def download_pplink(self):
        if not os.path.exists(self.filename):
            try:
                os.system(f"wget {self.pplink_url} -O {self.filename}")
            except Exception as e:
                print(e)

    def get_ens_df(self):
        keys = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame']

        with gzip.open(self.filename, 'rt', encoding='utf-8') as f:
            lines = f.readlines()

            ens_dict = {key: [None] * len(lines) for key in keys}

            with alive_bar(round(len(lines)), title='lines processed') as bar:
                for i in range(len(lines)):
                    bar()
                    line = lines[i]

                    if not line.startswith('#'):
                        parts = line.strip().split('\t')
                        attributes = parts[8].split('; ')

                        

                        for index, key in enumerate(keys):
                            ens_dict[key][i] = parts[index]


                        for attr in attributes:
                            key, value = attr.split(' ', 1)
                            key = key.strip('"')
                            value = value.strip('"')

                            if key not in ens_dict:
                                ens_dict[key] = [None] * len(lines)
                            ens_dict[key][i] = value

        df = pd.DataFrame(ens_dict)

        # must be 5 its the file readme  
        none_rows = df[df.isna().all(axis=1)]
        print(f"Lines with only None values: {none_rows.shape[0]}")

        df_cleaned = df.dropna(how='all')

        out_name = self.filename.split('.')[0]
        df_cleaned.to_csv(out_name+'_df.csv',index = False)
        print('GTF file parsed successfully')

def main():
    a = GTF_parsing()
    a.download_pplink()
    a.get_ens_df()

if __name__ == "__main__":
    main()
