#!/bin/bash

source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38
cd /home/ec2-user/SageMaker
git clone --recurse-submodules -j8 https://github.com/dglai/Graph-Neural-Networks-in-Life-Sciences.git
cd /home/ec2-user/SageMaker/Graph-Neural-Networks-in-Life-Sciences
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
source /home/ec2-user/anaconda3/bin/deactivate
​
