import argparse
import glob 
import time
import logging
import subprocess
import os
import pandas as pd
import multiprocessing as mp

#
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
logging.info(subprocess.call(f'ls -lR {INPUT_DIR}'.split()))
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

############################
# inference functions
############################
def runTextractNCompmed(objectName):
    input_object_prefix = opt.prefix
    output_dir = opt.output
    bucket_name = opt.bucket_name

    print(f'Object Name: {objectName}')
    
    #########################
    #
    # Run Textract
    #
    #########################
    lst_blocks_ = []
    response1 = textractC.start_document_text_detection(
       DocumentLocation = { 
          "S3Object":{ 
             "Bucket": bucket_name,
             "Name": objectName,
          }
        },
        #NotificationChannel= {
        #    "RoleArn": 'arn:aws:iam::<AWS Account ID>:role/service-role/AmazonSageMaker-ExecutionRole-20210215T135274',
        #    "SNSTopicArn": 'arn:aws:sns:us-east-1:<AWS Account ID>:AmazonTextract-kg-test-1'
        #},
    )

    while(1):
        response2 = textractC.get_document_text_detection(
            JobId=response1['JobId'],
            MaxResults=123,
        )
        time.sleep(5)

        if response2['JobStatus'] == 'SUCCEEDED':
            print(response2['JobStatus'])
            while(1):
                lst_blocks_ += response2["Blocks"]
                try: 
                    response2 = textractC.get_document_text_detection(
                                                                        JobId=response1['JobId'],
                                                                        MaxResults=123,
                                                                        NextToken=response2["NextToken"]
                                                                      )
                except Exception as e:
                    print(f"Exception when starting Textract JOB.: {e}")
                    break
            break
        else:
            print(response2['JobStatus'])

    lst_blocks = []
    for item in lst_blocks_:
        if item["BlockType"] == "LINE":
            lst_blocks.append(item)

    clusteredTexts = pd.DataFrame(lst_blocks).groupby(['Page']).Text.apply(lambda x: ' '.join(x)).reset_index()

    #########################
    #
    # Run CompMed
    #
    #########################
        
    response3 = compmedC.detect_entities(
        Text=list(clusteredTexts.Text)[0]  
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
        resultsDF['File'] = objectName
        resultsDF.drop(columns = ['index']).to_csv(objectName.replace(input_object_prefix, 
                                                                      output_dir).replace('pdf', 'csv'))
        print(objectName.replace(input_object_prefix, output_dir).replace('pdf', 'csv'))
        
    except Exception as e:
        print(e)

########################################
# Main
########################################
def main(opt):
    pdf_files_ = glob.glob(opt.input +'/*.pdf', recursive = True)
    pdf_files = [pdf.replace(opt.input, opt.prefix) for pdf in pdf_files_]
    
    # Small Sample
    #pdf_files = pdf_files[:10]
    
    with mp.Pool(num_cpu-1) as p:
        p.map(runTextractNCompmed, pdf_files)


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
    