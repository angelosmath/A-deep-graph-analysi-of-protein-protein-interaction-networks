#!/bin/bash
#SBATCH --job-name dgl
#SBATCH --partition=gpu
#SBATCH --mem=350G
#SBATCH --cpus-per-task=1
#SBATCH --output %j.out
#SBATCH --error %j.err
source /vast/groups/VEO/tools/miniconda3_2024/etc/profile.d/conda.sh


conda activate dgl


#conda env update --file /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/dgl.yml --prune
#
##echo "ucsc_data.py"
#cd /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/data/ucsc/
#python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/data/ucsc/ucsc_data.py
##echo "string"
#cd ../string
#python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/data/string/string_data.py
##echo "gtf parse"
#cd /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/gtf_parsing
#python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/gtf_parsing/gtf_parsing.py
##echo "ampping"
#cd /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/mapping
#python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/mapping/mapping.py
#echo "Preprocessing"
#cd /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/preprocessing
#pwd
#python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/preprocessing/preprocessing.py --path /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/mapping/STRING_gene.csv --folder preprocessing_results  --num_sampled_vectors 500
#echo "Pame"
cd /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/model/
echo "python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/model/trainer_neo_neo.py --training-graph /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/preprocessing/preprocessing_results/pkl/graph.dgl --test-size 0.2 --epochs 10 --lr 0.005 --train-batch-size 4096 --hidden-size 200 --exclude-edges 1 --output-folder 10_epochs"
python /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/model/trainer_neo_neo.py --training-graph /work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/preprocessing/preprocessing_results/pkl/graph.dgl --test-size 0.2 --epochs 10 --lr 0.005 --train-batch-size 4096 --hidden-size 200 --exclude-edges 1 --output-folder 10_epochs


echo "Done"