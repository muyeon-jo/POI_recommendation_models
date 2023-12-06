import time
import random
import numpy as np
import torch

def get_BPR_batch(X, test_negative, num_poi, batch_user_index):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(positives)
    histories = np.array([positives]).repeat(len(positives)*(negative_num+1),axis=0)

    negative = list(set(item_list)-set(positives) - set(test_negative[uid]))
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

def get_NAIS_batch_test(train_matrix, test_positive, test_negative, uid):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    history = train_matrix.getrow(uid).indices.tolist()
    negative = test_negative[uid]
    positive = test_positive[uid]
    histories = np.array([history]).repeat(len(positive)+len(negative),axis=0)

    data = np.concatenate((negative,positive))

    positive_label = np.array([1]).repeat(len(positive))
    negative_label = np.array([0]).repeat(len(negative))
    labels = np.concatenate((negative_label,positive_label))

    user_history = torch.LongTensor(histories).to(DEVICE)
    train_data = torch.LongTensor(data).to(DEVICE)
    train_label = torch.tensor(labels, dtype=torch.float32).to(DEVICE)

    return user_history, train_data, train_label

def get_NAIS_batch_region(train_matrix,test_negative, num_poi, uid, negative_num, businessRegionEmbedList):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(positives)
    histories = np.array([positives]).repeat(len(positives)*(negative_num+1),axis=0)

    negative = list(set(item_list)-set(positives) - set(test_negative[uid]))
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

def get_NAIS_batch_test_region(train_matrix, test_positive, test_negative, uid, businessRegionEmbedList):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    history = train_matrix.getrow(uid).indices.tolist()
    negative = test_negative[uid]
    positive = test_positive[uid]
    histories = np.array([history]).repeat(len(positive)+len(negative),axis=0)

    data = np.concatenate((negative,positive))

    positive_label = np.array([1]).repeat(len(positive))
    negative_label = np.array([0]).repeat(len(negative))
    labels = np.concatenate((negative_label,positive_label))

    user_history = histories
    train_data = data
    train_label = torch.tensor(labels,dtype=torch.float32).to(DEVICE)
    
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
