import os
import requests
import zipfile
from io import BytesIO
import pandas as pd
from itertools import combinations
from collections import defaultdict
import pickle
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# URL of the file
url = 'https://mips.helmholtz-muenchen.de/corum/download/releases/current/humanComplexes.txt.zip'
zip_filename = 'humanComplexes.txt.zip'
txt_filename = 'humanComplexes.txt'

# Check if the file already exists
if not os.path.exists(txt_filename):
    try:
        print("Downloading the file...")
        response = requests.get(url, verify=False)  # verify=False bypasses SSL certificate verification
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Download complete.")

        # Open the zip file from the response content
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Extract the .txt file from the zip
            z.extractall()
            print(f"Extracted file: {txt_filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"File {txt_filename} already exists. Skipping download.")

# Load the .txt file into a pandas DataFrame
if os.path.exists(txt_filename):
    df = pd.read_csv(txt_filename, sep='\t')
    df.to_csv('human_complexes.csv', index=False)

# Preprocess the Data
# Exclude rows with NaN values in 'subunits(Gene name)' or 'GO ID'
df = df.dropna(subset=['subunits(Gene name)', 'GO ID'])

# Remove duplicates
df = df.drop_duplicates(subset=['ComplexName'])

# Split the gene names and GO terms into lists
df['subunits_gene_list'] = df['subunits(Gene name)'].apply(lambda x: x.split(';'))
df['GO_terms_list'] = df['GO ID'].apply(lambda x: x.split(';'))

# Add a column to count the number of gene members in each complex
df['num_genes'] = df['subunits_gene_list'].apply(len)

# Optional: Filter complexes based on a specific number of gene members
min_genes = 40  # Set your minimum threshold here
max_genes = 20000000000000000000000000  # Set your maximum threshold here
df_filtered = df[(df['num_genes'] >= min_genes) & (df['num_genes'] <= max_genes)]

# Define a function to calculate Jaccard similarity
def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 1  # Both are empty, consider them as identical
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


similarities = []


for (i, row1), (j, row2) in combinations(df_filtered.iterrows(), 2):
    if row1['ComplexName'] == row2['ComplexName']:
        continue  # Skip comparison with itself
    
    gene_similarity = jaccard_similarity(row1['subunits_gene_list'], row2['subunits_gene_list'])
    go_similarity = jaccard_similarity(row1['GO_terms_list'], row2['GO_terms_list'])
    
    total_similarity = (gene_similarity + go_similarity) / 2  # Average similarity
    
    shared_genes_count = len(set(row1['subunits_gene_list']).intersection(set(row2['subunits_gene_list'])))
    shared_go_terms_count = len(set(row1['GO_terms_list']).intersection(set(row2['GO_terms_list'])))
    
    num_genes_1 = len(row1['subunits_gene_list'])
    num_genes_2 = len(row2['subunits_gene_list'])
    
    similarities.append((
        row1['ComplexName'], 
        row2['ComplexName'], 
        total_similarity, 
        num_genes_1, 
        num_genes_2, 
        shared_genes_count, 
        shared_go_terms_count
    ))

similarities_df = pd.DataFrame(similarities, columns=[
    'Complex1', 
    'Complex2', 
    'Similarity', 
    'NumGenes1', 
    'NumGenes2', 
    'SharedGenesCount', 
    'SharedGOTermsCount'
])

similarities_df = similarities_df.sort_values(by='Similarity', ascending=False)

most_close = similarities_df.head(2)
print("Two Most Close Complexes:")
print(most_close)

most_far = similarities_df.tail(2)
print("\nTwo Most Far Complexes:")
print(most_far)

mapping_path = '/home/angelosmath/MSc/thesis_final_last/preprocessing/preprocessing_results/pkl/node_id.pkl'  
with open(mapping_path, 'rb') as file:
    node_to_id = pickle.load(file)
print('Mapping file loaded!')

def prepare_tsne_plot(complex_pair, title):
    missing_genes = []
    d = defaultdict(list)

    for complex_name in complex_pair:
        gene_list = df_filtered[df_filtered['ComplexName'] == complex_name]['subunits_gene_list'].values
        if len(gene_list) > 0:
            gene_list = gene_list[0]
        else:
            print(f"Complex {complex_name} not found in the filtered DataFrame.")
            continue
        
        for gene in gene_list:
            if gene in node_to_id:
                d[complex_name].append(node_to_id[gene])
            else:
                missing_genes.append(gene)

    print(f"\nNumber of genes missing: {len(missing_genes)}")
    
    embeddings_path = 'embeddings.npy'  
    embeddings = np.load(embeddings_path)

    embeddings_d = defaultdict()

    for complex_name in list(d.keys()):
        if len(d[complex_name]) == 0:
            print(f"No valid genes found for complex {complex_name}.")
            continue
        selected_node_indices = d[complex_name]
        selected_embeddings = torch.tensor(embeddings[selected_node_indices], dtype=torch.float32)
        embeddings_d[complex_name] = selected_embeddings

    if len(embeddings_d) < 2:
        print("Not enough data for t-SNE. Need at least two complexes.")
        return

    combined_embeddings = np.vstack([embeddings.numpy() for embeddings in embeddings_d.values()])
    print(f"Shape of combined embeddings: {combined_embeddings.shape}")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings) - 1), n_iter=100000)
    tsne_combined = tsne.fit_transform(combined_embeddings)

    tsne_results = {}
    start = 0
    for complex_name, embeddings in embeddings_d.items():
        end = start + len(embeddings)
        tsne_results[complex_name] = tsne_combined[start:end]
        start = end
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tsne_results)))
    for (complex_name, result), color in zip(tsne_results.items(), colors):
        plt.scatter(result[:, 0], result[:, 1], label=complex_name, color=color)

    plt.legend()
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

prepare_tsne_plot(most_close[['Complex1', 'Complex2']].values[0], 't-SNE of Most Similar Complexes')
prepare_tsne_plot(most_far[['Complex1', 'Complex2']].values[0], 't-SNE of Most Distinct Complexes')
