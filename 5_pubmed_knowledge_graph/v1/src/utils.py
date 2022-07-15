import pandas as pd
import uuid
from glob import glob
from os import path as osp

def list_csvs(dir_path):
    path_expression = osp.join(dir_path, '*.csv')
    input_files = glob(path_expression)
    return input_files


def get_doc_record(df):
    doc_name = df.File[0].split('/')[-1]
    
    # create doc node
    doc_id = str(uuid.uuid4())
    doc_record = {
        'uid': doc_id, 'Name': doc_name
    }
    
    return doc_id, doc_record

# split input df into nodes & relationships
def split_df(df):
    node_idxs = df['RelationshipScore'].isna()
    node_df = df[node_idxs].copy()
    relationship_df = df[~node_idxs].copy()
    return node_df, relationship_df

# process nodes
def process_node_df(df, entity_thr, drop_duplicates_columns=('BeginOffset', 'EndOffset'), **kwargs):
    out_df = df[df.Score > entity_thr].drop('File', axis=1)
    out_df = out_df.drop_duplicates(subset=drop_duplicates_columns)
    return out_df


def get_cm_id_2_name(df):
    out_dict = df.set_index('Id').to_dict()['Text']
    return out_dict


# process relationships
def process_relationship_df(df, cm_id_2_uid, link_thr, **kwargs):
    
    # clean columns - this is a property of the edge to keep track of what file this is coming from
    df['File'] = df['File'].map(lambda x: x.split('/')[-1].split('.')[0])
    
    # filter out uncertain links
    out_df = df[df['RelationshipScore'] > link_thr].copy()
    
    # filter out uncertain nodes
    out_df['distId'] = out_df['distId'].astype(int)
    out_df = out_df[(out_df['distId'].isin(cm_id_2_uid)) & (out_df['Id'].isin(cm_id_2_uid))].copy()
    
    # process df
    out_df['from_node'] = [cm_id_2_uid[cm_id] for cm_id in out_df['distId']]
    out_df['to_node'] = [cm_id_2_uid[cm_id] for cm_id in out_df['Id']]
    out_df['uid'] = [str(uuid.uuid4()) for _ in range(len(out_df))]
    
    # select attributes
    cols = ['from_node', 'to_node', 'uid', 'RelationshipType', 'RelationshipScore', 'File']
    out_df = out_df[cols]
    
    return out_df

