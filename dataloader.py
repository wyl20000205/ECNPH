import sys
import numpy as np
from config import cfg
from torch.utils.data import Dataset,DataLoader
import os.path as osp
import h5py
import torch
import hdf5plugin

class BaseDataset(Dataset):
    def __init__(self,data):
        self.indexs,self.captions,self.labels = data
    def __len__(self):
        return  len(self.indexs)
    def __getitem__(self, index):
        return self.indexs[index], self.captions[index], self.labels[index], index
         
def dataloader(dataset_name='mirflickr25k'):
    np.random.seed(seed=cfg['seed'])
    with h5py.File(osp.join(cfg['project_root'],'data', f'{dataset_name}_256.h5'), 'r') as f:
         image_data = f['image'][:] # 20485 3 256 256
         caption_data = f['caption'][:] # bpe 20485 64
         label_data = f['label'][:]  # 20485 24
    print(f"loading dataset:{cfg['dataset']} image:{image_data.shape} caption:{caption_data.shape} label:{label_data.shape}")
    random_index = np.random.permutation(range(len(image_data))) 
    
    query_index = random_index[: cfg['num_query']]
    train_index = random_index[cfg['num_query']: cfg['num_query'] + cfg['num_train']] 
    retrieval_index = random_index[cfg['num_query']:] 
     
    train_indexs = image_data[train_index]
    train_captions = caption_data[train_index]
    train_labels = torch.Tensor(label_data[train_index]).float()
    
    query_indexs = image_data[query_index]
    query_captions = caption_data[query_index]
    query_labels =  torch.Tensor(label_data[query_index]).float()
    
    retrieval_indexs = image_data[retrieval_index]
    retrieval_captions = caption_data[retrieval_index]
    retrieval_labels = torch.Tensor(label_data[retrieval_index]).float()
    
    train_data = (train_indexs, train_captions, train_labels)
    query_data = (query_indexs, query_captions, query_labels)
    retrieval_data = (retrieval_indexs, retrieval_captions, retrieval_labels)
    
    train_dataset = BaseDataset(train_data)
    query_dataset = BaseDataset(query_data)
    retrieval_dataset = BaseDataset(retrieval_data)
    
    train_dataloader = DataLoader(train_dataset,batch_size=cfg['train_batch_size'],num_workers=cfg['num_workers'],pin_memory=True)
    query_dataloader = DataLoader(query_dataset,batch_size=cfg['query_batch_size'],num_workers=cfg['num_workers'],pin_memory=True)
    retrieval_dataloader = DataLoader(retrieval_dataset,batch_size=cfg['retrieval_batch_size'],num_workers=cfg['num_workers'],pin_memory=True)
    
    return (train_dataloader,query_dataloader,retrieval_dataloader),(train_labels,query_labels,retrieval_labels)
   

