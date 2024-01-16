import time
import random
import numpy as np
import torch

def get_BPR_batch(X, num_poi, batch_user_index):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    item_list = np.arange(num_poi).tolist()
    for uid in batch_user_index:
        poitives = X.getrow(uid).indices
        negative = list(set(item_list)-set(poitives))
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

def get_NAIS_batch(train_matrix, num_poi, uid, negative_num):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(positives)
    histories = np.array([positives]).repeat(len(positives)*(negative_num+1),axis=0)

    negative = list(set(item_list)-set(positives))
    random.shuffle(negative)

    negative = negative[:len(positives)*negative_num]
    negatives = np.array(negative).reshape([-1,negative_num])

    a= np.array(positives).reshape(-1,1)
    data = np.concatenate((a, negatives),axis=-1)
    data = data.reshape(-1)

    positive_label = np.array([1]).repeat(len(positives)).reshape(-1,1)
    negative_label = np.array([0]).repeat(len(positives)*negative_num).reshape(-1,negative_num)
    labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)

    user_history = torch.LongTensor(histories).to(DEVICE)
    train_data = torch.LongTensor(data).to(DEVICE)
    train_label = torch.tensor(labels,dtype=torch.float32).to(DEVICE)

    return user_history, train_data, train_label

def get_NAIS_batch_test(train_matrix, uid):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    history = train_matrix.getrow(uid).indices.tolist()
    data = list(set(range(train_matrix.shape[1]))-set(history))
    histories = np.array([history]).repeat(len(data),axis=0)

    labels = np.array([0]).repeat(len(data))

    user_history = torch.LongTensor(histories).to(DEVICE)
    train_data = torch.LongTensor(data).to(DEVICE)
    train_label = torch.tensor(labels, dtype=torch.float32).to(DEVICE)

    return user_history, train_data, train_label

def get_NAIS_batch_region(train_matrix, num_poi, uid, negative_num, businessRegionEmbedList):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(positives)
    histories = np.array([positives]).repeat(len(positives)*(negative_num+1),axis=0)

    negative = list(set(item_list)-set(positives))
    random.shuffle(negative)

    negative = negative[:len(positives)*negative_num]
    negatives = np.array(negative).reshape([-1,negative_num])

    a= np.array(positives).reshape(-1,1)
    data = np.concatenate((a, negatives),axis=-1)
    data = data.reshape(-1)

    positive_label = np.array([1]).repeat(len(positives)).reshape(-1,1)
    negative_label = np.array([0]).repeat(len(positives)*negative_num).reshape(-1,negative_num)
    labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)
    
    user_history = histories
    train_data = data
    train_label = torch.tensor(labels,dtype=torch.float32).to(DEVICE)
    
    user_history_region = []
    for i in positives:
        user_history_region.append(businessRegionEmbedList[i])
    
    user_history_region = np.array([user_history_region]).repeat(len(user_history),axis=0)

    train_data_region = []
    for i in train_data:
        train_data_region.append(businessRegionEmbedList[i])

    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)
    user_history_region=torch.LongTensor(user_history_region).to(DEVICE)
    train_data_region=torch.LongTensor(train_data_region).to(DEVICE)

    return user_history, train_data, train_label, user_history_region, train_data_region

def get_NAIS_batch_test_region(train_matrix, uid, businessRegionEmbedList):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    history = train_matrix.getrow(uid).indices.tolist()
    negative = list(set(range(train_matrix.shape[1]))-set(history))
    histories = np.array([history]).repeat(len(negative),axis=0)


    negative_label = np.array([0]).repeat(len(negative))

    user_history = histories
    train_data = negative
    train_label = torch.tensor(negative_label,dtype=torch.float32).to(DEVICE)
    
    user_history_region = []
    for i in history:
        user_history_region.append(businessRegionEmbedList[i])
    
    user_history_region = np.array([user_history_region]).repeat(len(user_history),axis=0)

    train_data_region = []
    for i in train_data:
        train_data_region.append(businessRegionEmbedList[i])

    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)
    user_history_region=torch.LongTensor(user_history_region).to(DEVICE)
    train_data_region=torch.LongTensor(train_data_region).to(DEVICE)
    
    return user_history, train_data, train_label, user_history_region, train_data_region

def get_GPR_batch(train_matrix, num_poi, uids, negative_num):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_positives = []
    train_negatives = []
    user_id = []
    for uid in uids:
        item_list = np.arange(num_poi).tolist()

        positives = train_matrix.getrow(uid).indices.tolist()
        random.shuffle(positives)
        
        user_id.extend(np.array([uid]).repeat(len(positives)).reshape(-1,1).tolist())
        negative = list(set(item_list)-set(positives))
        random.shuffle(negative)

        negative = negative[:len(positives)*negative_num]
        train_positives.extend(positives)
        train_negatives.extend(negative)
    
    train_positives = np.array(train_positives).reshape(-1,1)
    train_negatives = np.array(train_negatives).reshape(-1,negative_num)
    user_id = np.array(user_id).reshape(-1,1)

    train_positives = torch.LongTensor(train_positives).squeeze().to(DEVICE)
    train_negatives = torch.LongTensor(train_negatives).squeeze().to(DEVICE)
    user_id = torch.LongTensor(user_id).squeeze().to(DEVICE)

    return user_id, train_positives, train_negatives

def get_GPR_batch_test(train_matrix, uid):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datas = []
    history = train_matrix.getrow(uid).indices.tolist()
    negative = list(set(range(train_matrix.shape[1])) - set(history))
    user_id = []

    user_id.extend(np.array([uid]).repeat(len(negative)).tolist())
    datas.extend(negative)

    datas = torch.LongTensor(datas).to(DEVICE)
    user_id = torch.LongTensor(user_id).to(DEVICE)

    return user_id, datas

