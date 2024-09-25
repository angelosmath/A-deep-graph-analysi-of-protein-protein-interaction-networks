import os 
from alive_progress import alive_bar
import gzip
import pandas as pd


class STRING_pplink:

    def __init__(self):
        self.pplink_url = 'https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz' #'https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz'
        self.filename = '9606.protein.physical.links.v12.0.txt.gz'

    
    def download_pplink(self):
        if not os.path.exists(self.filename): 
            try:
                os.system(f"wget {self.pplink_url} -O {self.filename}")
            except Exception as e:
                print(e)

    def to_dataframe(self):

        with gzip.open(self.filename,'rt', encoding='utf-8') as f:
            
            lines = [line.strip('\n') for line in f]

            processed_lines = []

            with alive_bar(len(lines), title='lines processed') as bar:
                
                for line in lines:
                    bar()
                    if '.' in line:
                        elements = line.split()
                        
                        #remove species number before ensembl ID 
                        elements[0] = elements[0].split('.')[1]
                        elements[1] = elements[1].split('.')[1]
                    else:
                        elements = line.split()
                    processed_lines.append(elements)

        data = processed_lines[1:]
        columns = processed_lines[0]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('9606.physical.links.csv',index = False)
        

        
def main():
    gglink = STRING_pplink()
    gglink.download_pplink()
    gglink.to_dataframe()
    
    

if __name__ == "__main__":
    main()