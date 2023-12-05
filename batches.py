import time
import random
import numpy as np
import torch

def get_BPR_batch(X, test_negative, num_poi, batch_user_index):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    batch = []
    item_list = np.arange(num_poi).tolist()
    for uid in batch_user_index:
        poitives = X.getrow(uid).indices
        negative = list(set(item_list)-set(poitives) - set(test_negative[uid]))
        random.shuffle(negative)
        negative = negative[:len(poitives)]
        for i in range(len(poitives)):
            batch.append([uid,poitives[i],negative[i]])
    random.shuffle(batch)
    batch = np.array(batch).T
    user = torch.LongTensor(batch[0]).to(DEVICE)
    item_i = torch.LongTensor(batch[1]).to(DEVICE)
    item_j = torch.LongTensor(batch[2]).to(DEVICE)
    return user, item_i, item_j

def get_NAIS_batch(train_matrix,test_negative, num_poi, uid, negative_num):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    batch = []
    item_list = np.arange(num_poi).tolist()

    poitives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(poitives)

    negative = list(set(item_list)-set(poitives) - set(test_negative[uid]))
    random.shuffle(negative)

    negative = negative[:len(poitives)*negative_num]
    for i in range(len(poitives)):
        batch.append([poitives,poitives[i],1])
        for j in range(negative_num):
            batch.append([poitives,negative[i*negative_num + j],0])

    batch = np.array(batch,dtype=object).T
    user_history = torch.LongTensor(batch[0].tolist()).to(DEVICE)
    train_data = torch.LongTensor(batch[1].tolist()).to(DEVICE)
    train_label = torch.tensor(batch[2].tolist(),dtype=torch.float32).to(DEVICE)

    return user_history, train_data, train_label

def get_NAIS_batch_test(train_matrix, test_positive, test_negative, uid):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    batch = []
    history = train_matrix.getrow(uid).indices.tolist()
    negative = test_negative[uid]
    
    for j in range(len(negative)):
        batch.append([history,negative[j],0])

    for i in range(len(test_positive[uid])):
        batch.append([history,test_positive[uid][i],1])

    batch = np.array(batch,dtype=object).T
    user_history = torch.LongTensor(batch[0].tolist()).to(DEVICE)
    train_data = torch.LongTensor(batch[1].tolist()).to(DEVICE)
    train_label = torch.tensor(batch[2].tolist(), dtype=torch.float32).to(DEVICE)

    return user_history, train_data, train_label

def get_NAIS_batch_region(train_matrix,test_negative, num_poi, uid, negative_num, businessRegionEmbedList):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    batch = []
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(positives)

    negative = list(set(item_list)-set(positives) - set(test_negative[uid]))
    random.shuffle(negative)

    negative = negative[:len(positives)*negative_num]
    for i in range(len(positives)):
        batch.append([positives, positives[i],1])
    
        for j in range(negative_num):
            batch.append([positives  ,negative[i*negative_num + j],0])

    batch = np.array(batch,dtype=object).T
    user_history = batch[0].tolist()
    train_data = batch[1].tolist()
    train_label = torch.tensor(batch[2].tolist(),dtype=torch.float32).to(DEVICE)
    
    user_history_region = []
    for i in user_history:
        tmp = []
        for j in i:
            tmp.append(businessRegionEmbedList[j])
        user_history_region.append(tmp)

    train_data_region = []
    for i in train_data:
        train_data_region.append(businessRegionEmbedList[i])

    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)
    user_history_region=torch.LongTensor(user_history_region).to(DEVICE)
    train_data_region=torch.LongTensor(train_data_region).to(DEVICE)

    return user_history, train_data, train_label, user_history_region, train_data_region

def get_NAIS_batch_test_region(train_matrix, test_positive, test_negative, uid, businessRegionEmbedList):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    batch = [] 
    history = train_matrix.getrow(uid).indices.tolist()
    negative = test_negative[uid]
    
    for j in range(len(negative)):
        batch.append([history,negative[j],0])

    for i in range(len(test_positive[uid])):
        batch.append([history,test_positive[uid][i],1])

    batch = np.array(batch,dtype=object).T
    user_history = batch[0].tolist()
    train_data = batch[1].tolist()
    train_label = torch.tensor(batch[2].tolist(), dtype=torch.float32).to(DEVICE)
    
    user_history_region = []
    for i in user_history:
        tmp = []
        for j in i:
            tmp.append(businessRegionEmbedList[j])
        user_history_region.append(tmp)
    
    train_data_region = []
    for i in train_data:
        train_data_region.append(businessRegionEmbedList[i])
    
    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)
    user_history_region=torch.LongTensor(user_history_region).to(DEVICE)
    train_data_region=torch.LongTensor(train_data_region).to(DEVICE)
    
    return user_history, train_data, train_label, user_history_region, train_data_region

