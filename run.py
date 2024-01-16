from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os
import eval_metrics
import datasets
from batches import get_NAIS_batch_test_region,get_NAIS_batch_region,get_NAIS_batch_test,get_NAIS_batch,get_BPR_batch, get_GPR_batch
import torch
from powerLaw import PowerLaw, dist
from model import NAIS_basic, NAIS_regionEmbedding,NAIS_region_distance_Embedding,BPR, NAIS_region_distance_disentangled_Embedding, NAIS_distance_Embedding, GPR
import time
import random
import math
import validation as val
import multiprocessing as mp
import torch.cuda as T
import pickle
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
class Args:
    def __init__(self):
        self.lr = 0.01# learning rate
        self.lamda = 0.0 # model regularization rate
        self.batch_size = 4096 # batch size for training
        self.epochs = 40 # training epoches
        self.topk = 50 # compute metrics@top_k
        self.factor_num = 16 # predictive factors numbers in the model
        self.hidden_dim = 64 # predictive factors numbers in the model
        self.num_ng = 4 # sample negative items for training
        self.out = True # save model or not
        self.beta = 0.5
        self.powerlaw_weight = 0.2

def train_NAIS(train_matrix, test_positive, val_positive, dataset):
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num

    model = NAIS_basic(num_items, args.factor_num,args.factor_num,0.5).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)

        for buid in idx:
            
            user_history, train_data, train_label = get_NAIS_batch(train_matrix,num_items,buid,args.num_ng)
            optimizer.zero_grad() # 그래디언트 초기화
            prediction = model(user_history, train_data)
            loss = model.loss_func(prediction,train_label)
            loss.backward() # 역전파 및 그래디언트 계산

            train_loss += loss.item()
            
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            start_time = int(time.time())
            precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.NAIS_validation(model,args,num_users,test_positive,val_positive,train_matrix,k_list)
            
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
def train_NAIS_region(train_matrix, test_positive, val_positive, dataset):

    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(dataset.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]
    
    model = NAIS_regionEmbedding(num_items, args.factor_num,args.hidden_dim,0.5,region_num).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            optimizer.zero_grad() # 그래디언트 초기화
            user_history , train_data, train_label, user_history_region, train_data_region = get_NAIS_batch_region(train_matrix, num_items, buid, args.num_ng, businessRegionEmbedList)
            
            prediction = model(user_history, train_data, user_history_region, train_data_region)
            #if buid == 0:
            #    print(f"prediction : {prediction.shape}, {prediction}")
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            start_time = int(time.time())
            precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.NAIS_region_validation(model,args,num_users,test_positive,val_positive,train_matrix,businessRegionEmbedList,k_list)
            
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
def train_NAIS_region_distance(train_matrix, test_positive, val_positive, dataset):

    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_distance"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_distance"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    poi_coos = G.poi_coos 
    # latlon_mat = lat_lon_mat(num_items,poi_coos)
    # pickle_save(latlon_mat,dataset.directory_path+"latlon_mat.pkl")
    latlon_mat = pickle_load(dataset.directory_path+"latlon_mat.pkl")
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(dataset.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]

    model = NAIS_region_distance_Embedding(num_items, args.factor_num, args.factor_num, args.beta, region_num,1).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            user_history , train_data, train_label, user_history_region, train_data_region = get_NAIS_batch_region(train_matrix, num_items, buid, args.num_ng, businessRegionEmbedList)
            history_pois = [i for i in user_history[0].tolist()] # 방문한 데이터
            target_pois = [i for i in train_data.tolist()] # 타겟 데이터
            
            target_lat_long = []
            for poi1 in target_pois: #타겟 데이터에 대해서 거리 계산 batch_size
                hist = latlon_mat[[poi1 for i in history_pois],history_pois]
                target_lat_long.append(hist.tolist())
            target_lat_long=torch.tensor(target_lat_long,dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad() # 그래디언트 초기화

            prediction = model(user_history, train_data, user_history_region, train_data_region, target_lat_long)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            start_time = int(time.time())
            precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.NAIS_region_distance_validation(model,args,num_users,test_positive,val_positive,train_matrix,businessRegionEmbedList, latlon_mat,k_list)
            end_time = int(time.time())
            print("eval time: {} sec".format(end_time-start_time))
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
def train_NAIS_region_disentangled_distance(train_matrix, test_positive, val_positive, dataset):

    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_distance_disentangled"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_distance_disentangled"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    poi_coos = G.poi_coos 
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(dataset.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]

    model = NAIS_region_distance_disentangled_Embedding(num_items, args.factor_num, args.factor_num, args.beta, region_num,1).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            user_history , train_data, train_label, user_history_region, train_data_region = get_NAIS_batch_region(train_matrix, num_items, buid, args.num_ng, businessRegionEmbedList)
            history_pois = [(poi_coos[i][0], poi_coos[i][1]) for i in user_history[0]] # 방문한 데이터
            target_pois = [(poi_coos[i][0], poi_coos[i][1]) for i in train_data] # 타겟 데이터
            
            target_dist = []
            for poi1 in target_pois: #타겟 데이터에 대해서 거리 계산 batch_size
                hist = []
                for poi2 in history_pois:#history_size
                    hist.append(dist(poi1, poi2))
                target_dist.append(hist)
            target_dist=torch.tensor(target_dist,dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad() # 그래디언트 초기화

            prediction = model(user_history, train_data, user_history_region, train_data_region, target_dist)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            start_time = int(time.time())
            val_precision, val_recall, val_hit = val.NAIS_region_distance_validation(model,args,num_users,val_positive,train_matrix,businessRegionEmbedList, poi_coos,True,[10])
            
            if(max_recall < val_recall[0]):
                max_recall = val_recall[0]
                torch.save(model, model_directory+"/model")
                precision, recall, hit = val.NAIS_region_distance_validation(model,args,num_users,test_positive,train_matrix,businessRegionEmbedList, poi_coos, False,k_list)
                f=open(result_directory+"/results.txt","w")
                f.write("epoch:{}\n".format(e))
                f.write("@k: " + str(k_list)+"\n")
                f.write("prec:" + str(precision)+"\n")
                f.write("recall:" + str(recall)+"\n")
                f.write("hit:" + str(hit)+"\n")
                f.close()
            end_time = int(time.time())
            print("eval time: {} sec".format(end_time-start_time))
def train_NAIS_distance(train_matrix, test_positive, test_negative, val_positive, dataset):

    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_distance"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"NAIS_region_distance"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    poi_coos = G.poi_coos 
    region_num = datasets.get_region_num(dataset.directory_path)
    # latlon_mat = lat_lon_mat(num_items,poi_coos)
    # pickle_save(latlon_mat,dataset.directory_path+"latlon_mat.pkl")
    latlon_mat = pickle_load(dataset.directory_path+"latlon_mat.pkl")
    with open(dataset.directory_path+"poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]

    model = NAIS_distance_Embedding(num_items, args.factor_num, args.factor_num, args.beta, region_num,1).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            user_history , train_data, train_label, user_history_region, train_data_region = get_NAIS_batch_region(train_matrix, num_items, buid, args.num_ng, businessRegionEmbedList)
            history_pois = [i for i in user_history[0].tolist()] # 방문한 데이터
            target_pois = [i for i in train_data.tolist()] # 타겟 데이터
            
            target_lat_long = []
            for poi1 in target_pois: #타겟 데이터에 대해서 거리 계산 batch_size
                hist = latlon_mat[[poi1 for i in history_pois],history_pois]
                target_lat_long.append(hist.tolist())
            target_lat_long=torch.tensor(target_lat_long,dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad() # 그래디언트 초기화

            prediction = model(user_history, train_data, user_history_region, train_data_region, target_lat_long)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            start_time = int(time.time())
            precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.NAIS_region_distance_validation(model,args,num_users,test_positive,val_positive,train_matrix,businessRegionEmbedList, latlon_mat,k_list)
            
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

def train_BPR(train_matrix, test_positive, val_positive, dataset):
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"BPR"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"BPR"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    ########################### CREATE MODEL #################################
    # BPR 모델 생성
    model = BPR(num_users, num_items, args.factor_num).to(DEVICE)

    # 옵티마이저 생성 (SGD 사용)
    optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.lamda)

    ########################### TRAINING #####################################        
    print("start")
    for epoch in range(args.epochs):
        model.train()  # 모델을 학습 모드로 설정
        start_time = time.time() # 시작시간 기록            
        # train_loader.dataset.ng_sample() # negative 예제 샘플링
        idx = np.arange(num_users).tolist()
        random.shuffle(idx)
        n = math.ceil(len(idx)/args.batch_size)

        batch_user_index = []
        for i in range(n):
            if(i == n-1):
                batch_user_index.append(idx[args.batch_size*i:])
            else:
                batch_user_index.append(idx[args.batch_size*i:args.batch_size*(i+1)])
        lo = 0.0
        # for user, item_i, item_j in train_loader:
        for buid in batch_user_index:
            user, item_i, item_j = get_BPR_batch(train_matrix,num_items,buid)
            optimizer.zero_grad() # 그래디언트 초기화
            prediction_i, prediction_j = model(user, item_i, item_j)
            loss = - (prediction_i - prediction_j).sigmoid().log().sum() # BPR 손실 함수 계산
            lo += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        
        elapsed_time = time.time() - start_time  # 소요시간 계산
        print(lo)
        print(f"Epoch : {epoch}, Used_Time : {elapsed_time}")
        if epoch %20 == 0:
            model.eval() # 모델을 평가 모드로 설정
            with torch.no_grad():
                start_time = int(time.time())
                val_precision, val_recall, val_hit = val.BPR_validation(model,args,num_users,val_positive,True,[10])
                
                if(max_recall < val_recall[0]):
                    max_recall = val_recall[0]
                    torch.save(model, model_directory+"/model")
                    precision, recall, hit= val.BPR_validation(model,args,num_users,test_positive,False,k_list)
                    alpha = args.powerlaw_weight

                    # recommended_list = []
                    # recommended_list_g = []
                    # for user_id in range(num_users):
                    #     user_tensor = torch.LongTensor([user_id] * (len(test_negative[user_id])+len(test_positive[user_id]))).to(DEVICE)
                    #     target_list = test_negative[user_id]+test_positive[user_id]
                    #     target_tensor = torch.LongTensor(target_list).to(DEVICE)

                    #     prediction, _ = model(user_tensor, target_tensor,target_tensor)

                    #     _, indices = torch.topk(prediction, args.topk)
                    #     recommended_list.append([target_list[i] for i in indices])

                    #     G_score = normalize([G.predict(user_id,poi_id) for poi_id in target_list])
                    #     G_score = torch.tensor(np.array(G_score)).to(DEVICE)
                    #     prediction = (1-alpha)*prediction + alpha * G_score

                    #     _, indices = torch.topk(prediction, args.topk)
                    #     recommended_list_g.append([target_list[i] for i in indices])


                    # precision, recall, hit = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
                    # precision_g, recall_g, hit_g = eval_metrics.evaluate_mp(test_positive,recommended_list_g,k_list)
                    f=open(result_directory+"/results.txt","w")
                    f.write("epoch:{}\n".format(epoch))
                    f.write("@k: " + str(k_list)+"\n")
                    f.write("prec:" + str(precision)+"\n")
                    f.write("recall:" + str(recall)+"\n")
                    f.write("hit:" + str(hit)+"\n")

                    # f.write("distance weight: {}\n".format(alpha))

                    # f.write("prec:" + str(precision_g)+"\n")
                    # f.write("recall:" + str(recall_g)+"\n")
                    # f.write("hit:" + str(hit_g)+"\n")
                    f.close()
                end_time = int(time.time())
                print("eval time: {} sec".format(end_time-start_time))

def train_GPR(train_matrix, test_positive, val_positive, dataset):
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"GPR"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"GPR"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]
    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    dist_mat = distance_mat(num_items, G.poi_coos)
    pickle_save(dist_mat,dataset.directory_path+"dist_mat.pkl")
    dist_mat = pickle_load(dataset.directory_path+"dist_mat.pkl")
    dist_mat = torch.tensor(dist_mat,dtype=torch.float32).to(DEVICE)
    model = GPR(num_users, num_items, args.factor_num, 2, dataset.POI_POI_Graph,dist_mat, dataset.user_POI_Graph).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)
    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        
        random.shuffle(idx)
        user_id, train_positives, train_negatives = get_GPR_batch(train_matrix,num_items,idx,args.num_ng)
        aaaa = 1
        batches = []
        nnn = int(len(user_id)/50)
        for i in range(49):
            batches.append((nnn*(i),nnn*(i+1)))
        
        batches.append((nnn*49,len(user_id)))
        for idx_range in batches:
            optimizer.zero_grad() 
            
            rating_ul, rating_ul_prime, e_ij_hat = model(user_id[idx_range[0]:idx_range[1]], train_positives[idx_range[0]:idx_range[1]], train_negatives[idx_range[0]:idx_range[1]])
            loss = model.loss_function(rating_ul, rating_ul_prime, e_ij_hat)
            loss.backward()

            train_loss += loss.item()
            print(loss.item())
            optimizer.step() 
            aaaa+=1
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() 
        with torch.no_grad():
            start_time = int(time.time())
            precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.GPR_validation(model,args,num_users,test_positive,val_positive,train_matrix,k_list)
            end_time = int(time.time())
            print("eval time: {} sec".format(end_time-start_time))
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

def main():
    print("data loading")
    # dataset_ = datasets.Dataset(3725,10768,"./data/Tokyo/")
    # train_matrix, test_positive, val_positive, place_coords = dataset_.generate_data(0)
    # pickle_save((train_matrix, test_positive, val_positive, place_coords,dataset_),"dataset_Tokyo.pkl")
    train_matrix, test_positive, val_positive, place_coords, dataset_ = pickle_load("dataset_Tokyo.pkl")
    print("train data generated")
    # datasets.get_region(place_coords,200,dataset_.directory_path)
    # datasets.get_region_num(dataset_.directory_path)
    print("geo file generated")
    
    G.fit_distance_distribution(train_matrix, place_coords)
    
    print("train start")
    # train_NAIS_distance(train_matrix, test_positive, test_negative, val_positive, val_negative, dataset_)
    train_NAIS_region_distance(train_matrix, test_positive, val_positive, dataset_)
    # train_NAIS_region_disentangled_distance(train_matrix, test_positive, test_negative, val_positive, val_negative, dataset_)
    
    # train_NAIS_region(train_matrix, test_positive, val_positive, dataset_)
    # train_NAIS(train_matrix, test_positive, val_positive, dataset_)
    # train_BPR(train_matrix, test_positive, test_negative, val_positive, val_negative, dataset_)

if __name__ == '__main__':
    G = PowerLaw()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    main()
