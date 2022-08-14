import argparse
import glob 
import time
import logging
import subprocess
import os
import pandas as pd
import multiprocessing as mp

#
import json
import logging
import os
import sys
import gzip
from xml.etree import ElementTree
from ftplib import FTP
import re
import shutil
from functools import partial

url = "ftp.ncbi.nlm.nih.gov"
ftp_path = "/pubmed/baseline/"
reg_ex = "pubmed22n\\d{4}\.xml\\.gz$"

local_path = './downloaded'
re_obj = re.compile(reg_ex)


import boto3
s3r = boto3.resource('s3')
 
textractC = boto3.client('textract', region_name=os.environ['MY_REGION'])
compmedC = boto3.client('comprehendmedical', region_name=os.environ['MY_REGION'])
#

INPUT_DIR = "/opt/ml/processing/input" # change this for testing locally
OUTPUT_DIR = "/opt/ml/processing/output" # change this for testing locally 
MODEL_DIR = "/opt/ml/model" # change this for testing locally
SOURCE_DIR = "/opt/ml/code" # change this for testing locally

logging.basicConfig(level=logging.DEBUG)
#logging.info(subprocess.call(f'ls -lR {INPUT_DIR}'.split()))
logging.info(subprocess.call(f'ls -lR {SOURCE_DIR}'.split()))

num_cpu = mp.cpu_count()
print(f'Number of CPU : {num_cpu}')

############################
# helper functions
############################
def get_host_id():
    host = os.environ['HOSTNAME']
    id_ = "".join(host.split('.')[0].split('-')[-2:])
    return id_


def groupConsequtiveSingleTokens(l): 
    clusters = l.copy()
    c = 0
    for i in l.index:
        if not (i > l.index[0] and l[i-1] == 1):
            c += 1
        clusters[i] = c
    return clusters

def get_text(element, path):
    ele = element.find(path)
    if ele is not None:
        return ele.text
    else:
        return None

############################
# inference functions
############################
def getPubMedNCompmed(filename):
    input_object_prefix = opt.prefix
    output_dir = opt.output
    bucket_name = opt.bucket_name
    
    #print(f'Object Name: {objectName}')
    
    #########################
    #
    # Get Pubmed
    #
    #########################

    
    
    ftp = FTP(url)
    ftp.login()
    ftp.cwd(ftp_path)
    #print("Downloading {} ..".format(filename))
    local_filename = os.path.join(local_path, filename)
    with open(local_filename, 'wb') as f_:
        ftp.retrbinary('RETR ' + filename, f_.write)
    with gzip.open(local_filename, 'rb') as f_in:
        tmp_file = "{}.tmp".format(local_filename)
        with open(tmp_file, 'wb+') as f_out:
            shutil.copyfileobj(f_in, f_out)
            f_out.seek(0)
            events = ElementTree.iterparse(f_out, events=("start", "end"))
            _, root = next(events)  # Grab the root element.

            result = []
            i_count = 0

            for event, elem in events:
                pmid = get_text(elem, "MedlineCitation/PMID")
                abstract = get_text(elem, "MedlineCitation/Article/Abstract/AbstractText")

                if abstract is not None and pmid is not None:
                    resultsDF = detectEntities(abstract, pmid, output_dir)
                    i_count += 1
                    #if i_count > 10:
                    #    break
                else:
                    continue
            
            os.remove(local_filename)
            os.remove(tmp_file)
                                    

#########################
#
# Run CompMed
#
#########################
        
def detectEntities(abstract, pmid, output_dir):

    response3 = compmedC.detect_entities(
        Text=abstract  
    )
    
    lst_attributes = []
    for sublist in response3['Entities']:
        try:
            if 'Attributes' in sublist.keys():
                resultsDF__ = pd.DataFrame(sublist['Attributes'])
                resultsDF__['distId'] = int(sublist['Id'])
                lst_attributes.append(resultsDF__)
        except Exception as e:
            print(e)
            
    try:
        resultsDF = pd.DataFrame(response3['Entities'])#.reset_index()

        df_middle = pd.concat(lst_attributes)

        df_middle = df_middle[['Id',
         'BeginOffset',
         'EndOffset',
         'Score',
         'Text',
         'Category',
         'Type',
         'Traits']]

        df_middle = df_middle.drop_duplicates(subset=['BeginOffset', 'EndOffset'], keep='first')
        df_middle['Attributes'] =''
        df_middle = df_middle[pd.DataFrame(response3['Entities']).columns.tolist()]

        resultsDF = pd.concat([resultsDF, df_middle, pd.concat(lst_attributes)]).reset_index()
        resultsDF['File'] = pmid
        
        #resultsDF.drop(columns = ['index']).to_csv(objectName.replace(input_object_prefix, 
        #                                                              output_dir).replace('pdf', 'csv'))
        resultsDF.drop(columns = ['index']).to_csv(output_dir + '/' + pmid + '.csv')                                                              
        return resultsDF
    except Exception as e:
        print(e)
        return pd.DataFrame([])

########################################
# Main
########################################
def main(opt):

    re_obj = re.compile(reg_ex)
    os.makedirs(local_path, exist_ok=True)

    ftp_ = FTP(url)
    ftp_.login()
    ftp_.cwd(ftp_path)
    file_names = ftp_.nlst()
    
    gz_files = list(filter(lambda f: re_obj.match(f) is not None, file_names))
    gz_files.sort()
    
    # Small Sample
    #gz_files = pdf_files[:10]
   
    #func = partial(getPubMedNCompmed, opt, ftp)
    
    with mp.Pool(num_cpu-1) as p:
        p.map(getPubMedNCompmed, gz_files)
        #p.map(runTextractNCompmed, pdf_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.input = INPUT_DIR
    opt.output = OUTPUT_DIR    
    opt.source = SOURCE_DIR
    opt.region = os.environ['MY_REGION']
    opt.bucket_name = os.environ['BUCKET_NAME']
    opt.prefix = os.environ['S3_PREFIX']
    opt.host_id = get_host_id()
    opt.model_dir = MODEL_DIR

    t1 = time.time()
    main(opt)
    print(f"Time taken: {time.time() - t1}")             
    