from argparse import ArgumentParser
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import eval_metrics
import datasets
from batches import *
from powerLaw import PowerLaw, dist
from model import *
import time
import random
import math
import validation as val
import multiprocessing as mp
import torch.cuda
import torch
import pickle
from torchmetrics.functional.pairwise import pairwise_manhattan_distance
from haversine import haversine, haversine_vector

from torch.utils.viz._cycles import warn_tensor_cycles

def pickle_load(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_save(data, name):
    with open(name, 'wb') as f:
	    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# parser = ArgumentParser(description="SAE-NAD")
# parser.add_argument('-e', '--epoch', type=int, default=60, help='number of epochs for GAT')
# parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size for training')
# parser.add_argument('--alpha', type=float, default=2.0, help='the parameter of the weighting function')
# parser.add_argument('--epsilon', type=float, default=1e-5, help='the parameter of the weighting function')
# parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
# parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
# parser.add_argument('-att', '--num_attention', type=int, default=20, help='the number of dimension of attention')
# parser.add_argument('--inner_layers', nargs='+', type=int, default=[200, 50, 200], help='the number of latent factors')
# parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5, help='the dropout probability')
# parser.add_argument('-seed', type=int, default=0, help='random state to split the data')
# args = parser.parse_args()
def save_result(path, filename, result, setting):
    """
    path(str): 저장경로
    filename(str): 저장할 파일 이름
    result(list): 모델 성능 측정 결과 recall, precision, hit 순서
    setting(dict): 모델에 적용한 파라미터 (파라미터명,적용값)
    """
    with open(path+filename) as f:
        for k,v in setting:
            f.write(str(k)+":"+str(v)+"\n")
        f.write("\n")
        f.write("recall, precision, hitrate\n")

        for li in result:
            for i in li:
                f.write(str(i)+"\t")
            f.write("\n")

def distance_mat(poi_num,poi_coos):
    dist_mat = np.zeros([poi_num,poi_num])
    for i in range(poi_num):
        for j in range(poi_num):
            dist_mat[i][j] = min(max(0.01,dist(poi_coos[i], poi_coos[j])), 100)
    
    return dist_mat
def lat_lon_mat(poi_num,poi_coos):
    dist_mat = np.zeros([poi_num,poi_num,2])
    for i in range(poi_num):
        for j in range(poi_num):
            dist_mat[i][j][0] = abs(poi_coos[i][0]-poi_coos[j][0])
            dist_mat[i][j][1] = abs(poi_coos[i][1]-poi_coos[j][1])
    
    return dist_mat
def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores
class trainDataset(Dataset):
    def __init__(self, train_matrix,region_data, negative_num=4):
        self.train_matrix = train_matrix #csr_matrix
        self.user_num = train_matrix.shape[0]
        self.poi_num = train_matrix.shape[1]
        self.item_list = np.arange(self.train_matrix.shape[1]).tolist()

        self.POI_region_mapping = region_data
        
        li = self.train_matrix.sum(axis = 0)
        self.item_visit_num = np.array(li.data.tolist()).reshape(-1)
        
        self.positive_list = []
        self.positive_freq_list = []
        for i in range(self.user_num):
            self.positive_list.append(train_matrix.getrow(i).indices.tolist())
            self.positive_freq_list.append(self.train_matrix.getrow(i).data.tolist())

        self.negative_num = negative_num
    def __len__(self):
        return self.train_matrix.shape[0]
    
    def __getitem__(self, uid):
        positives = self.positive_list[uid]
        pos_visit_num = np.array(self.positive_freq_list[uid])
        visit_rate = torch.from_numpy(pos_visit_num / self.item_visit_num[positives])
        histories = np.array([positives]).repeat(len(positives)*(self.negative_num+1),axis=0)

        negative = list(set(self.item_list)-set(positives))
        random.shuffle(negative)
        negative = negative[:len(positives)*self.negative_num]
        negatives = np.array(negative).reshape([-1,self.negative_num])

        a = np.array(positives).reshape(-1,1)
        data = np.concatenate((a, negatives),axis=-1)
        data = data.reshape(-1)

        positive_label = np.array([1.0],dtype=np.float32).repeat(len(positives)).reshape(-1,1)
        negative_label = np.array([0.0],dtype=np.float32).repeat(len(positives)*self.negative_num).reshape(-1,self.negative_num)
        labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)
        
        user_history = torch.from_numpy(histories)
        train_data = torch.from_numpy(data)
        train_label = torch.from_numpy(labels)
        # train_label = labels
        user_history_region = self.POI_region_mapping[positives]
        
        user_history_region = torch.from_numpy(np.array([user_history_region]).repeat(len(user_history),axis=0))

        train_data_region = torch.from_numpy(self.POI_region_mapping[train_data.tolist()])


        return user_history, train_data, train_label, user_history_region, train_data_region, visit_rate

class testDataset(Dataset):
    def __init__(self, train_matrix,region_data):
        self.train_matrix = train_matrix #csr_matrix
        self.region_data = region_data
        li = self.train_matrix.sum(axis = 0)
        
        self.item_visit_num = np.array(li.data.tolist()).reshape(-1)
    def __len__(self):
        return self.train_matrix.shape[0]
    
    def __getitem__(self, uid):
        history = train_matrix.getrow(uid).indices.tolist()
        negative = np.array(list(set(range(train_matrix.shape[1]))-set(history)))

        pos_visit_num = np.array(self.train_matrix.getrow(uid).data.tolist(),dtype=np.float64)
        visit_rate = pos_visit_num / self.item_visit_num[history]

        histories = np.array([history]).repeat(len(negative),axis=0)


        negative_label = np.array([0]).repeat(len(negative))

        user_history = histories
        train_data = negative
        train_label = negative_label
        
        user_history_region = self.region_data[history]
        
        user_history_region = np.array([user_history_region]).repeat(len(user_history),axis=0)

        train_data_region = self.region_data[train_data]

        return user_history, train_data, train_label, user_history_region, train_data_region, visit_rate

class trainDataset2(Dataset):
    def __init__(self, train_matrix,region_data, poi_coos,negative_num=4):
        self.train_matrix = train_matrix #csr_matrix
        self.user_num = train_matrix.shape[0]
        self.poi_num = train_matrix.shape[1]
        self.item_list = np.arange(self.train_matrix.shape[1]).tolist()
        self.poi_coos = poi_coos
        self.POI_region_mapping = region_data
        
        li = self.train_matrix.sum(axis = 0)
        self.item_visit_num = np.array(li.data.tolist()).reshape(-1)
        
        self.positive_list = []
        self.positive_freq_list = []
        for i in range(self.user_num):
            self.positive_list.append(train_matrix.getrow(i).indices.tolist())
            self.positive_freq_list.append(self.train_matrix.getrow(i).data.tolist())
        t1 = poi_coos.tolist()
        self.POI_POI_distance = torch.from_numpy(haversine_vector(t1,t1,comb=True))

        self.negative_num = negative_num
    def __len__(self):
        return self.train_matrix.shape[0]
    
    def __getitem__(self, uid):
        positives = self.positive_list[uid]
        pos_visit_num = np.array(self.positive_freq_list[uid])
        visit_rate = torch.from_numpy(pos_visit_num / self.item_visit_num[positives])
        histories = np.array([positives]).repeat(len(positives)*(self.negative_num+1),axis=0)

        negative = list(set(self.item_list)-set(positives))
        random.shuffle(negative)
        negative = negative[:len(positives)*self.negative_num]
        negatives = np.array(negative).reshape([-1,self.negative_num])

        a = np.array(positives).reshape(-1,1)
        data = np.concatenate((a, negatives),axis=-1)
        data = data.reshape(-1)

        positive_label = np.array([1.0],dtype=np.float32).repeat(len(positives)).reshape(-1,1)
        negative_label = np.array([0.0],dtype=np.float32).repeat(len(positives)*self.negative_num).reshape(-1,self.negative_num)
        labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)
        
        user_history = torch.from_numpy(histories)
        train_data = torch.from_numpy(data)
        train_label = torch.from_numpy(labels)
        # train_label = labels
        user_history_region = self.POI_region_mapping[positives]
        user_history_region = torch.from_numpy(np.array([user_history_region]).repeat(len(user_history),axis=0))

        train_data_region = torch.from_numpy(self.POI_region_mapping[train_data.tolist()])
        a = torch.LongTensor(positives)
        dist_mat = self.POI_POI_distance[:,train_data][a]

        return user_history, train_data, train_label, user_history_region, train_data_region, visit_rate, dist_mat, uid

class testDataset2(Dataset):
    def __init__(self, train_matrix,region_data,poi_coos):
        self.train_matrix = train_matrix #csr_matrix
        self.region_data = region_data
        li = self.train_matrix.sum(axis = 0)
        self.poi_coos = poi_coos
        self.item_visit_num = np.array(li.data.tolist()).reshape(-1)
        t1 = poi_coos.tolist()
        self.POI_POI_distance = torch.from_numpy(haversine_vector(t1,t1,comb=True))

    def __len__(self):
        return self.train_matrix.shape[0]
    
    def __getitem__(self, uid):
        history = train_matrix.getrow(uid).indices.tolist()
        negative = np.array(list(set(range(train_matrix.shape[1]))-set(history)))

        pos_visit_num = np.array(self.train_matrix.getrow(uid).data.tolist(),dtype=np.float64)
        visit_rate = pos_visit_num / self.item_visit_num[history]

        histories = np.array([history]).repeat(len(negative),axis=0)


        negative_label = np.array([0]).repeat(len(negative))

        user_history = histories
        train_data = negative
        train_label = negative_label
        
        user_history_region = self.region_data[history]
        
        user_history_region = np.array([user_history_region]).repeat(len(user_history),axis=0)

        train_data_region = self.region_data[train_data]
        dist_mat = self.POI_POI_distance[:,train_data][history]
        return user_history, train_data, train_label, user_history_region, train_data_region, visit_rate, dist_mat, uid

class trainDataset3(Dataset):
    def __init__(self, train_matrix,region_data, negative_num=1):
        self.train_matrix = train_matrix #csr_matrix
        self.user_num = train_matrix.shape[0]
        self.poi_num = train_matrix.shape[1]
        self.item_list = np.arange(self.train_matrix.shape[1]).tolist()

        self.POI_region_mapping = region_data
        
        li = self.train_matrix.sum(axis = 0)
        self.item_visit_num = np.array(li.data.tolist()).reshape(-1)
        
        self.positive_list = []
        self.positive_freq_list = []
        for i in range(self.user_num):
            self.positive_list.append(train_matrix.getrow(i).indices.tolist())
            self.positive_freq_list.append(self.train_matrix.getrow(i).data.tolist())

        self.negative_num = negative_num
    def __len__(self):
        return self.train_matrix.shape[0]
    
    def __getitem__(self, uid):
        positives = self.positive_list[uid]

        negatives = list(set(self.item_list)-set(positives))
        random.shuffle(negatives)
        negatives = negatives[:len(positives)*self.negative_num]

        a = np.array(positives)
        ids = np.array(uid).repeat(len(positives))
        negatives = torch.tensor(negatives)
        positives = torch.from_numpy(a)
        negatives = negatives
        # train_label = labels

        return positives, negatives, ids

class testDataset3(Dataset):
    def __init__(self, train_matrix,region_data):
        self.train_matrix = train_matrix #csr_matrix
        self.region_data = region_data
        li = self.train_matrix.sum(axis = 0)
        
        self.item_visit_num = np.array(li.data.tolist()).reshape(-1)
    def __len__(self):
        return self.train_matrix.shape[0]
    
    def __getitem__(self, uid):
        history = train_matrix.getrow(uid).indices.tolist()
        negative = np.array(list(set(range(train_matrix.shape[1]))-set(history)))

        pos_visit_num = np.array(self.train_matrix.getrow(uid).data.tolist(),dtype=np.float64)
        visit_rate = pos_visit_num / self.item_visit_num[history]

        histories = np.array([history]).repeat(len(negative),axis=0)


        negative_label = np.array([0]).repeat(len(negative))

        user_history = histories
        train_data = negative
        train_label = negative_label
        
        user_history_region = self.region_data[history]
        
        user_history_region = np.array([user_history_region]).repeat(len(user_history),axis=0)

        train_data_region = self.region_data[train_data]

        return user_history, train_data, train_label, user_history_region, train_data_region, visit_rate

class Args:
    def __init__(self):
        self.lr = 0.001# learning rate            
        self.lamda = 1e-04 # model regularization rate
        self.batch_size = 4096 # batch size for training
        self.epochs = 100 # training epoches
        self.topk = 50 # compute metrics@top_k
        self.factor_num = 128 # predictive factors numbers in the model
        self.hidden_dim = 128 # predictive factors numbers in the model
        self.num_ng = 4 # sample negative items for training
        self.beta = 0.5
        self.powerlaw_weight = 0.2
        self.sampling_ratio = 0.2
        self.gglr_control = 0.2
        self.scaling = 10

def train_NAIS_new(train_matrix, test_positive, val_positive, dataset):
    print(train_matrix.shape)
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_new"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_new"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    with open(result_directory+"/setting.txt","w") as setting_f:
        setting_f.write("lr:{}\n".format(str(args.lr)))
        setting_f.write("lamda:{}\n".format(str(args.lamda)))
        setting_f.write("epochs:{}\n".format(str(args.epochs)))
        setting_f.write("factor_num:{}\n".format(str(args.factor_num)))
        setting_f.write("hidden_dim:{}\n".format(str(args.hidden_dim)))
        setting_f.write("num_ng:{}\n".format(str(args.num_ng)))
    num_users = dataset.user_num
    num_items = dataset.poi_num
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(dataset_.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]
    businessRegionEmbedList = np.array(businessRegionEmbedList)
    training_data = trainDataset(train_matrix,businessRegionEmbedList,args.num_ng)
    test_data = testDataset(train_matrix,businessRegionEmbedList)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = New1(num_items, args.factor_num,args.hidden_dim,0.5,region_num)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        for i,(user_history , train_data, train_label, user_history_region, train_data_region, visit_rate ) in enumerate(train_dataloader):
            user_history=user_history.squeeze(0).to(DEVICE)
            train_data=train_data.squeeze(0).to(DEVICE)
            train_label=train_label.squeeze(0).to(DEVICE)
            user_history_region=user_history_region.squeeze(0).to(DEVICE)
            train_data_region=train_data_region.squeeze(0).to(DEVICE)
            visit_rate=visit_rate.squeeze(0).to(DEVICE)
            optimizer.zero_grad() # 그래디언트 초기화
            # user_history , train_data, train_label, user_history_region, train_data_region, visit_rate = get_New1_batch(train_matrix, num_items, buid, args.num_ng, businessRegionEmbedList)
        
            prediction = model(user_history, train_data, user_history_region, train_data_region, visit_rate)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
            
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        if (e+1)%10 == 0:
            model.eval() # 모델을 평가 모드로 설정
            with torch.no_grad():
                start_time = int(time.time())
                # recommended_list = []
                # for user_id in range(num_users):
                #     user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)

                #     prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate)
                #     # loss = model.loss_func(prediction,train_label)
                #     # train_loss += loss.item()
                
                #     _, indices = torch.topk(prediction, args.topk)
                #     recommended_list.append([target_list[i].item() for i in indices])
                #     del (user_history, target_list, train_label, user_history_region, train_data_region, visit_rate) # 학습 데이터 삭제
                #     torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
                # precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
                # precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)

                recommended_list = []
                for user_id,(user_history, target_list, _, user_history_region, train_data_region, visit_rate) in enumerate(test_dataloader):
                    # user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)
                    user_history=user_history.squeeze(0).to(DEVICE)
                    target_list=target_list.squeeze(0).to(DEVICE)
                    
                    user_history_region=user_history_region.squeeze(0).to(DEVICE)
                    train_data_region=train_data_region.squeeze(0).to(DEVICE)
                    visit_rate=visit_rate.squeeze(0).to(DEVICE)

                    prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate)
                
                    _, indices = torch.topk(prediction, args.topk)
                    recommended_list.append([target_list[i].item() for i in indices])
                    del (user_history, target_list, user_history_region, train_data_region, visit_rate) # 학습 데이터 삭제
                    torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
                precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
                precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
                # precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.New1_validation(model,args,num_users,test_positive,val_positive,train_matrix,businessRegionEmbedList,k_list)
                
                if(max_recall < recall_v[1]):
                    max_recall = recall_v[1]
                    torch.save(model, model_directory+"/model")
                    f=open(result_directory+"/results.txt","w")
                    f.write("epoch:{}\n".format(e))
                    f.write("@k: " + str(k_list)+"\n")
                    f.write("prec:" + str(precision_t)+"\n")
                    f.write("recall:" + str(recall_t)+"\n")
                    f.write("hit:" + str(hit_t)+"\n")
                    f.close()
                end_time = int(time.time())
                print("eval time: {} sec".format(end_time-start_time))


def train_NAIS_new2(train_matrix, test_positive, val_positive, dataset):
    print(train_matrix.shape)
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"_new2"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"_new2"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    with open(result_directory+"/setting.txt","w") as setting_f:
        setting_f.write("lr:{}\n".format(str(args.lr)))
        setting_f.write("lamda:{}\n".format(str(args.lamda)))
        setting_f.write("epochs:{}\n".format(str(args.epochs)))
        setting_f.write("factor_num:{}\n".format(str(args.factor_num)))
        setting_f.write("hidden_dim:{}\n".format(str(args.hidden_dim)))
        setting_f.write("num_ng:{}\n".format(str(args.num_ng)))
    num_users = dataset.user_num
    num_items = dataset.poi_num
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(dataset_.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]
    businessRegionEmbedList = np.array(businessRegionEmbedList)
    training_data = trainDataset2(train_matrix,businessRegionEmbedList,np.array(dataset.place_coos),args.num_ng)
    test_data = testDataset2(train_matrix,businessRegionEmbedList,np.array(dataset.place_coos))
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = New2(num_items, args.factor_num,args.hidden_dim,0.5,region_num,num_users)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        for i,(user_history , train_data, train_label, user_history_region, train_data_region, visit_rate, dist_mat, uid) in enumerate(train_dataloader):
            user_history=user_history.squeeze(0).to(DEVICE)
            train_data=train_data.squeeze(0).to(DEVICE)
            train_label=train_label.squeeze(0).to(DEVICE)
            user_history_region=user_history_region.squeeze(0).to(DEVICE)
            train_data_region=train_data_region.squeeze(0).to(DEVICE)
            visit_rate=visit_rate.squeeze(0).to(DEVICE)
            dist_mat = dist_mat.squeeze(0).to(DEVICE)
            uid =uid.squeeze(0).to(DEVICE)

            optimizer.zero_grad() # 그래디언트 초기화
            # user_history , train_data, train_label, user_history_region, train_data_region, visit_rate = get_New1_batch(train_matrix, num_items, buid, args.num_ng, businessRegionEmbedList)
        
            prediction = model(user_history, train_data, user_history_region, train_data_region, visit_rate, dist_mat, uid)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
            
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        if (e+1)%10 == 0:
            model.eval() # 모델을 평가 모드로 설정
            with torch.no_grad():
                start_time = int(time.time())
                # recommended_list = []
                # for user_id in range(num_users):
                #     user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)

                #     prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate)
                #     # loss = model.loss_func(prediction,train_label)
                #     # train_loss += loss.item()
                
                #     _, indices = torch.topk(prediction, args.topk)
                #     recommended_list.append([target_list[i].item() for i in indices])
                #     del (user_history, target_list, train_label, user_history_region, train_data_region, visit_rate) # 학습 데이터 삭제
                #     torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
                # precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
                # precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)

                recommended_list = []
                for user_id,(user_history, target_list, _, user_history_region, train_data_region, visit_rate, dist_mat, uid) in enumerate(test_dataloader):
                    # user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)
                    user_history=user_history.squeeze(0).to(DEVICE)
                    target_list=target_list.squeeze(0).to(DEVICE)
                    
                    user_history_region=user_history_region.squeeze(0).to(DEVICE)
                    train_data_region=train_data_region.squeeze(0).to(DEVICE)
                    visit_rate=visit_rate.squeeze(0).to(DEVICE)
                    dist_mat = dist_mat.squeeze(0).to(DEVICE)
                    uid =uid.squeeze(0).to(DEVICE)

                    prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate,dist_mat,uid)
                
                    _, indices = torch.topk(prediction, args.topk)
                    recommended_list.append([target_list[i].item() for i in indices])
                    del (user_history, target_list, user_history_region, train_data_region, visit_rate,dist_mat,uid) # 학습 데이터 삭제
                    torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
                precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
                precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
                # precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.New1_validation(model,args,num_users,test_positive,val_positive,train_matrix,businessRegionEmbedList,k_list)
                
                if(max_recall < recall_v[1]):
                    max_recall = recall_v[1]
                    torch.save(model, model_directory+"/model")
                    f=open(result_directory+"/results.txt","w")
                    f.write("epoch:{}\n".format(e))
                    f.write("@k: " + str(k_list)+"\n")
                    f.write("prec:" + str(precision_t)+"\n")
                    f.write("recall:" + str(recall_t)+"\n")
                    f.write("hit:" + str(hit_t)+"\n")
                    f.close()
                end_time = int(time.time())
                print("eval time: {} sec".format(end_time-start_time))


def train_NAIS_new3(train_matrix, test_positive, val_positive, dataset):
    print(train_matrix.shape)
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"_new3"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"_new3"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    with open(result_directory+"/setting.txt","w") as setting_f:
        setting_f.write("lr:{}\n".format(str(args.lr)))
        setting_f.write("lamda:{}\n".format(str(args.lamda)))
        setting_f.write("epochs:{}\n".format(str(args.epochs)))
        setting_f.write("factor_num:{}\n".format(str(args.factor_num)))
        setting_f.write("hidden_dim:{}\n".format(str(args.hidden_dim)))
        setting_f.write("num_ng:{}\n".format(str(args.num_ng)))
    num_users = dataset.user_num
    num_items = dataset.poi_num
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(dataset_.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]
    businessRegionEmbedList = np.array(businessRegionEmbedList)
    training_data = trainDataset3(train_matrix,businessRegionEmbedList,1)
    test_data = testDataset3(train_matrix,businessRegionEmbedList)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = New3(num_users,num_items, args.factor_num)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = np.arange(num_users).tolist()
        random.shuffle(idx)
        n = math.ceil(len(idx)/args.batch_size)

        batch_user_index = []
        for i in range(n):
            if(i == n-1):
                batch_user_index.append(idx[args.batch_size*i:])
            else:
                batch_user_index.append(idx[args.batch_size*i:args.batch_size*(i+1)])

    
        # for user, item_i, item_j in train_loader:
        for buid in batch_user_index:
            user, item_i, item_j = get_BPR_batch(train_matrix,num_items,buid)
            optimizer.zero_grad() # 그래디언트 초기화
            prediction_i, prediction_j = model(user, item_i, item_j)
            loss = model.bpr_loss(prediction_i,prediction_j) # BPR 손실 함수 계산
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
            
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        if (e+1)%10 == 0:
            model.eval() # 모델을 평가 모드로 설정
            with torch.no_grad():
                start_time = int(time.time())
                # recommended_list = []
                # for user_id in range(num_users):
                #     user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)

                #     prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate)
                #     # loss = model.loss_func(prediction,train_label)
                #     # train_loss += loss.item()
                
                #     _, indices = torch.topk(prediction, args.topk)
                #     recommended_list.append([target_list[i].item() for i in indices])
                #     del (user_history, target_list, train_label, user_history_region, train_data_region, visit_rate) # 학습 데이터 삭제
                #     torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
                # precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
                # precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)

                recommended_list = []
                for user_id,(user_history, target_list, _, user_history_region, train_data_region, visit_rate, dist_mat, uid) in enumerate(test_dataloader):
                    # user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)
                    user_history=user_history.squeeze(0).to(DEVICE)
                    target_list=target_list.squeeze(0).to(DEVICE)
                    
                    user_history_region=user_history_region.squeeze(0).to(DEVICE)
                    train_data_region=train_data_region.squeeze(0).to(DEVICE)
                    visit_rate=visit_rate.squeeze(0).to(DEVICE)
                    dist_mat = dist_mat.squeeze(0).to(DEVICE)
                    uid =uid.squeeze(0).to(DEVICE)

                    prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate,dist_mat,uid)
                
                    _, indices = torch.topk(prediction, args.topk)
                    recommended_list.append([target_list[i].item() for i in indices])
                    del (user_history, target_list, user_history_region, train_data_region, visit_rate,dist_mat,uid) # 학습 데이터 삭제
                    torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
                precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
                precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
                # precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.New1_validation(model,args,num_users,test_positive,val_positive,train_matrix,businessRegionEmbedList,k_list)
                
                if(max_recall < recall_v[1]):
                    max_recall = recall_v[1]
                    torch.save(model, model_directory+"/model")
                    f=open(result_directory+"/results.txt","w")
                    f.write("epoch:{}\n".format(e))
                    f.write("@k: " + str(k_list)+"\n")
                    f.write("prec:" + str(precision_t)+"\n")
                    f.write("recall:" + str(recall_t)+"\n")
                    f.write("hit:" + str(hit_t)+"\n")
                    f.close()
                end_time = int(time.time())
                print("eval time: {} sec".format(end_time-start_time))

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #reference cycle detector
    # torch.cuda.memory._record_memory_history()

    # DEVICE = 'cpu'
    G = PowerLaw()
    print("data loading")
    # dataset_ = datasets.Dataset(3725,10768,"./data/Tokyo/")
    # train_matrix, test_positive, val_positive, place_coords = dataset_.generate_data(0)
    # pickle_save((train_matrix, test_positive, val_positive, place_coords,dataset_),"dataset_Tokyo.pkl")
    train_matrix, test_positive, val_positive, place_coords, dataset_ = pickle_load("dataset_Tokyo.pkl")
    print("train data generated")
    # datasets.get_region(place_coords,300,dataset_.directory_path)
    # datasets.get_region_num(dataset_.directory_path)
    print("geo file generated")
    
    G.fit_distance_distribution(train_matrix, np.array(place_coords))
    
    print("train start")
    
    train_NAIS_new3(train_matrix, test_positive, val_positive, dataset_)

    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    # G = PowerLaw()
    # print("data loading")
    # dataset_ = datasets.Dataset(15359,14586,"./data/Yelp/")
    # train_matrix, test_positive, val_positive, place_coords = dataset_.generate_data(0)
    # pickle_save((train_matrix, test_positive, val_positive, place_coords,dataset_),"dataset_Yelp.pkl")
    # train_matrix, test_positive, val_positive, place_coords, dataset_ = pickle_load("dataset_Yelp.pkl")
    # print("train data generated")
    # datasets.get_region(place_coords,300,dataset_.directory_path)
    # datasets.get_region_num(dataset_.directory_path)
    # print("geo file generated")
    
    # G.fit_distance_distribution(train_matrix, np.array(place_coords))
    # print("train start")
    
    # train_NAIS_new(train_matrix, test_positive, val_positive, dataset_)

    # G = PowerLaw()
    # print("data loading")
    # dataset_ = datasets.Dataset(6638,21102,"./data/NewYork/")
    # train_matrix, test_positive, val_positive, place_coords = dataset_.generate_data(0)
    # pickle_save((train_matrix, test_positive, val_positive, place_coords,dataset_),"dataset_NewYork.pkl")
    # train_matrix, test_positive, val_positive, place_coords, dataset_ = pickle_load("dataset_NewYork.pkl")
    # print("train data generated")
    # datasets.get_region(place_coords,300,dataset_.directory_path)
    # datasets.get_region_num(dataset_.directory_path)
    # print("geo file generated")
    
    # G.fit_distance_distribution(train_matrix, np.array(place_coords))
    # print("train start")
    
    # train_NAIS_new(train_matrix, test_positive, val_positive, dataset_)