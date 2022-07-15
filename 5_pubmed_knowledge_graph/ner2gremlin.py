import argparse
import glob 
import time
import logging
import subprocess
import os

## Config
from pathlib2 import Path
from datetime import datetime
import src
from src.knowledge_graph_transformer import knowledgeGraphTransformer
from src.utils import list_csvs
import config

#
import boto3
s3r = boto3.resource('s3')

#

INPUT_DIR = "/opt/ml/processing/input" # change this for testing locally
OUTPUT_DIR = "/opt/ml/processing/output" # change this for testing locally 
MODEL_DIR = "/opt/ml/model" # change this for testing locally
SOURCE_DIR = "/opt/ml/code" # change this for testing locally

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call(f'ls -lR {INPUT_DIR}'.split()))
logging.info(subprocess.call(f'ls -lR {SOURCE_DIR}'.split()))

print('passed')
############################
# helper functions
############################
def get_host_id():
    host = os.environ['HOSTNAME']
    id_ = "".join(host.split('.')[0].split('-')[-2:])
    return id_

############################
# inference functions
############################
class ConfigClass:
    def __init__(self,config):
        print('This class is working')
        self.INPUT_DIR = INPUT_DIR ##
        self.LOCAL_OUTPUT = OUTPUT_DIR ##
        self.S3_OUTPUT = f's3://{config.bucket_name}/{config.prefix}/gremlin-data-dummy/'
        self.NOW = config.NOW
        self.kwargs = config.kwargs

########################################
# Main
########################################
#def main(opt):
#    config.INPUT_DIR = opt.input
#    config.LOCAL_DIR = opt.output
#    
#    input_files = list_csvs(config.INPUT_DIR)
#
#    kgt = knowledgeGraphTransformer(config)
#    kgt.prepare_neptune_data_from_files(input_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    print("parser")
    opt = parser.parse_args()

    opt.input = INPUT_DIR
    opt.output = OUTPUT_DIR    
    opt.source = SOURCE_DIR
    opt.model_dir = MODEL_DIR

    print("parser from envs")
    opt.region = os.environ['MY_REGION']
    opt.bucket_name = os.environ['BUCKET_NAME']
    opt.prefix = os.environ['S3_PREFIX']

    print("get_host_id")
    opt.host_id = get_host_id()

    print("time")
    t1 = time.time()
    opt.NOW = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    opt.kwargs = config.kwargs
    
    input_files = list_csvs(INPUT_DIR)

    print("Rewrite config")
    config_ = ConfigClass(opt)
    print(config_)
    print(config_.INPUT_DIR)
    print(config_.LOCAL_OUTPUT)
    
    print("StartKG")
    kgt = knowledgeGraphTransformer(config_)
    kgt.prepare_neptune_data_from_files(input_files)    

    print(f"Time taken: {time.time() - t1}")             
    