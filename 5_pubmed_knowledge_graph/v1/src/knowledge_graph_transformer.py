import pandas as pd
import uuid
from pathlib2 import Path
from os import path as osp
import subprocess
from tqdm import tqdm


from base.base_graph_object_collections import NodeList, EdgeList
from utils import (
    get_doc_record,
    split_df,
    process_node_df,
    get_cm_id_2_name,
    process_relationship_df
)

class knowledgeGraphTransformer(object):
    
    def __init__(self, config):
        '''
        Initialize the configuration file. Example of required fields in v1/config.py.
        
        Args
            config (module): configuration module. Example in `v1/config.py`
            
        Returns
            None
        '''
        self.kwargs = config.kwargs
        self.time_stamp = config.NOW
        self.local_output = config.LOCAL_OUTPUT
        self.s3_output = config.S3_OUTPUT

        # --- Outputs ---
        self.node_root = osp.join(self.local_output, f'nodes/')
        self.edge_root = osp.join(self.local_output, f'edges/')
        Path(self.node_root).mkdir(parents=True, exist_ok=True)
        Path(self.edge_root).mkdir(parents=True, exist_ok=True)
        

        # --- Initializations ---
        self.node_lists = {
            'doc': NodeList(node_label='doc', 
                             output_file=osp.join(self.node_root, f'doc_nodes.csv'),
                             unique_attr='Name')
        }

        self.edge_lists = {
            'doc2text': EdgeList(edge_label='contains_text', 
                                 output_file=osp.join(self.edge_root, f'doc2text_edges.csv'),
                                 unique_attr='Name')
        }
        
        
    def _extract_from_file(self, file_path):
        '''
        Process a single filepath by extracting nodes and relationship from it.
        
        Args
            input_files (list): Input filepaths to be processed and converted into graph objects.   
            
        Returns
            None
        '''
        input_df = pd.read_csv(file_path, index_col='Unnamed: 0')
        
        doc_id, doc_record = get_doc_record(input_df)
        self.node_lists['doc'].record_object(doc_record)

        # split into node & relationship
        node_df, relationship_df = split_df(input_df)

        # proc node df
        proc_node_df = process_node_df(node_df, **self.kwargs)

        cm_id_2_uid = {}
        for node_type, df_type in proc_node_df.groupby('Type'):

            if node_type not in self.node_lists:
                self.node_lists[node_type] = NodeList(node_label=node_type, 
                                                 output_file=osp.join(self.node_root, f'{node_type}_nodes.csv'),
                                                 unique_attr='Name')

            node_type_list = self.node_lists[node_type]

            # nodes concept
            type_df2 = df_type.drop('Type', axis=1).rename({'Text':'Name'}, axis=1)
            type_df2['uid'] = [node_type_list.elements.get(name, {'~id': str(uuid.uuid4())})['~id'] 
                               for name in type_df2['Name']]
            node_records = type_df2[['uid', 'Name', 'Category']].to_dict(orient='records')
            node_type_list.record_list_of_objects(node_records)
            type_cm_id2uid = {cm_id:node_type_list[name]['~id'] 
                              for cm_id, name in zip(type_df2.Id, type_df2.Name)}
            cm_id_2_uid.update(type_cm_id2uid)

            # edge doc -> concept
            type_df3 = type_df2.rename({'uid':'to_node'}, axis=1)
            type_df3['from_node'] = [doc_id]*len(type_df3)
            type_df3['uid'] = [str(uuid.uuid4()) for _ in range(len(type_df3))]
            
            # Once we agree on doc->concept edge properties sub-select columns here
            edge_records = type_df3.to_dict(orient='records')
            self.edge_lists['doc2text'].record_list_of_objects(edge_records)


        proc_relationship_df = process_relationship_df(relationship_df, cm_id_2_uid, **self.kwargs)

        for edge_type, edge_type_df in proc_relationship_df.groupby('RelationshipType'):

            if edge_type not in self.edge_lists:
#                 unique_att set to 'uid' so relationships coming from different files are added without being 
#                 considered duplicates based on origin/destination nodes
                self.edge_lists[edge_type] = EdgeList(edge_label=edge_type, 
                                                 output_file=osp.join(self.edge_root, f'{edge_type}_edges.csv'),
                                                 unique_attr='uid')

            edge_records = edge_type_df.drop('RelationshipType', axis=1).to_dict(orient='records')
            self.edge_lists[edge_type].record_list_of_objects(edge_records)
        
    
    def _construct_graph_objects(self, input_files):
        '''
        Construct graph objects from input files
        
        Args
            input_files (list): Input filepaths to be processed and converted into graph objects.   
            
        Returns
            None.
        '''
        for file_path in tqdm(input_files):
            self._extract_from_file(file_path)

        
    def _write_locally(self):
        '''
        Write memory-stored in LOCAL_OUTPUT (config file).
        
        Args
            None
            
        Returns
            None
        '''
        for node_list in self.node_lists.values():
            node_list.to_csv()
        print(f'Nodes stored at {self.node_root}')

        for edge_list in self.edge_lists.values():
            edge_list.to_csv()
        print(f'Edges stored at {self.node_root}')
        
        
    def _push_to_s3(self):
        '''
        Push graph objects in LOCAL_OUTPUT to S3_OUTPUT bucket (config file).
        
        Args
            None
            
        Returns
            None
            
        '''
        ### 08/20/2021 Tats : SGM processing check
        upload_process = subprocess.run(['aws', 's3', 'cp', '--recursive', self.local_output, self.s3_output],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("\n---- Uploading Objects.... ")
        print("stdout: ", upload_process.stdout)
        print("stderr: ", upload_process.stderr)
        
    
    def prepare_neptune_data_from_files(self, input_files):
        '''
        Prepare Gremlin bulck upload folder in S3 folder specified in config file.
        
        Args
            input_files (list): List of documents to be processed.
            
        Returns
            None
        '''
        
        print('Creating objects...')
        self._construct_graph_objects(input_files)
        print('Storing gremlin files locally...')
        self._write_locally()
        print('Pushing gremlin files to S3...')
        ### 08/20/2021 Tats : SGM processing check
        #self._push_to_s3()
    