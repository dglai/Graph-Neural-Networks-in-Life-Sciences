from pathlib2 import Path
from datetime import datetime

NOW = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# inputs
INPUT_DIR = '/home/ec2-user/SageMaker/data/cm_data_0804'
OUTPUT_DIR = f'/home/ec2-user/SageMaker/data/gremlin_output/08_26/{NOW}'
S3_OUTPUT = f's3://knoma-hcls-mlsl-v-team/gremlin_data/{NOW}'

# processing configs
kwargs = {
    'entity_thr': .5,
    'link_thr': .5
}

