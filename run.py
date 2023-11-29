from argparse import ArgumentParser
import numpy as np

import eval_metrics
import datasets
from batches import get_NAIS_batch_test_region,get_NAIS_batch_region,get_NAIS_batch_test,get_NAIS_batch,get_BPR_batch
import torch
from powerLaw import PowerLaw, dist
from model import NAIS_basic, NAIS_regionEmbedding,NAIS_region_distance_Embedding,BPR
import time
import random
import math
import multiprocessing as mp
if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


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

def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


class Args:
    def __init__(self):
        self.lr = 0.01 # learning rate
        self.lamda = 0.00 # model regularization rate
        self.batch_size = 4096 # batch size for training
        self.epochs = 150 # training epoches
        self.topk = 50 # compute metrics@top_k
        self.factor_num = 16 # predictive factors numbers in the model
        self.region_embed_size=152
        self.num_ng = 4 # sample negative items for training
        self.out = True # save model or not
        self.beta = 0.5
        self.powerlaw_weight = 0.2

def train_NAIS(train_matrix, test_positive, test_negative, dataset):

    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num

    
    model = NAIS_basic(num_items, args.factor_num,args.factor_num,0.5).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    recall = []
    prec = []
    hit = []

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        # random.shuffle(idx)

        for buid in idx:
            user_history, train_data, train_label = get_NAIS_batch(train_matrix,test_negative,num_items,buid,args.num_ng)
            model.zero_grad() # 그래디언트 초기화
            prediction = model(user_history, train_data)
            # print(train_label)
            # print(prediction)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        if e%30==0:
            model.eval() # 모델을 평가 모드로 설정
            alpha = args.powerlaw_weight
            recommended_list = []
            recommended_list_g = []
            start_time = int(time.time())
            for user_id in range(num_users):
                user_history, target_list, train_label = get_NAIS_batch_test(train_matrix,test_positive,test_negative,user_id)

                prediction = model(user_history, target_list)

                _, indices = torch.topk(prediction, args.topk)
                recommended_list.append([target_list[i].item() for i in indices])

                G_score = normalize([G.predict(user_id,poi_id) for poi_id in target_list])
                G_score = torch.tensor(np.array(G_score)).to(DEVICE)
                prediction = (1-alpha)*prediction + alpha * G_score

                _, indices = torch.topk(prediction, args.topk)
                recommended_list_g.append([target_list[i].item() for i in indices])
            end_time = int(time.time())
            print("time: {}".format(end_time-start_time))
            k_list=[5, 10, 15, 20,25,30]
            precision, recall, hit = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
            precision_g, recall_g, hit_g = eval_metrics.evaluate_mp(test_positive,recommended_list_g,k_list)

    return recall , precision, hit
def train_NAIS_region(train_matrix, test_positive, test_negative, dataset):
    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(".\data\Yelp\poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]
    
    model = NAIS_regionEmbedding(num_items, args.factor_num,args.factor_num*2,0.5,region_num).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    recall = []
    prec = []
    hit = []

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            user_history , train_data, train_label, user_history_region, train_data_region = get_NAIS_batch_region(train_matrix, test_negative, num_items, buid, args.num_ng, businessRegionEmbedList)
            model.zero_grad() # 그래디언트 초기화
            prediction = model(user_history, train_data, user_history_region, train_data_region)
            #if buid == 0:
            #    print(f"prediction : {prediction.shape}, {prediction}")
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss/len(idx)))
        
        if(e%10 == 0):
            model.eval() # 모델을 평가 모드로 설정
            alpha = args.powerlaw_weight
            recommended_list = []
            recommended_list_g = []
            start_time = time.time()
            for user_id in range(num_users):
                user_history, target_list, train_label, user_history_region, train_data_region = get_NAIS_batch_test_region(train_matrix,test_positive,test_negative,user_id, businessRegionEmbedList)

                prediction = model(user_history, target_list, user_history_region, train_data_region)

                _, indices = torch.topk(prediction, args.topk)
                recommended_list.append([target_list[i].item() for i in indices])

                G_score = normalize([G.predict(user_id,poi_id) for poi_id in target_list])
                G_score = torch.tensor(np.array(G_score)).to(DEVICE)
                prediction = (1-alpha)*prediction + alpha * G_score

                _, indices = torch.topk(prediction, args.topk)
                recommended_list_g.append([target_list[i].item() for i in indices])
            end_time = int(time.time())
            print("time: {}".format(end_time-start_time))
            k_list=[5, 10, 15, 20,25,30]
            precision, recall, hit = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
            precision_g, recall_g, hit_g = eval_metrics.evaluate_mp(test_positive,recommended_list_g,k_list)
            

    return recall , precision, hit
def train_NAIS_region_distance(train_matrix, test_positive, test_negative, dataset):
    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    poi_coos = G.poi_coos 
    region_num = datasets.get_region_num(dataset.directory_path)
    with open(".\data\Yelp\poi_region_sorted.txt", 'r') as file:
        # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        businessRegionEmbedList = [int(line.split('\t')[1].strip()) for line in file.readlines()]

    model = NAIS_region_distance_Embedding(num_items, args.factor_num, args.factor_num*2, args.beta, args.region_embed_size, region_num).to(DEVICE)

    # 옵티마이저 생성 (adagrad 사용)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    recall = []
    prec = []
    hit = []

    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            user_history , train_data, train_label, user_history_region, train_data_region = get_NAIS_batch_region(train_matrix, test_negative, num_items, buid, args.num_ng, businessRegionEmbedList)
            history_pois = [(poi_coos[i][0], poi_coos[i][1]) for i in user_history[0]] # 방문한 데이터
            target_pois = [(poi_coos[i][0], poi_coos[i][1]) for i in train_data] # 타겟 데이터
            
            target_dist = []
            for poi1 in target_pois: #타겟 데이터에 대해서 거리 계산
                avg = 0
                for poi2 in history_pois:
                    avg += dist(poi1, poi2)
                avg /= len(history_pois)
                target_dist.append(avg)
            target_dist=torch.tensor(target_dist,dtype=torch.float32).to(DEVICE)
            model.zero_grad() # 그래디언트 초기화
            prediction = model(user_history, train_data, user_history_region, train_data_region, target_dist)
            loss = model.loss_func(prediction,train_label)
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss/len(idx)))
        
        if e % 10 == 0:
            model.eval() # 모델을 평가 모드로 설정
            alpha = args.powerlaw_weight
            recommended_list = []
            recommended_list_g = []
            start_time = time.time()
            for user_id in range(num_users):
                user_history, target_list, train_label, user_history_region, train_data_region = get_NAIS_batch_test_region(train_matrix,test_positive,test_negative,user_id, businessRegionEmbedList)        
                history_pois = [(poi_coos[i][0], poi_coos[i][1]) for i in user_history[0]] # 방문한 데이터
                target_pois = [(poi_coos[i][0], poi_coos[i][1]) for i in target_list] # 타겟 데이터
                target_dist = []
                for poi1 in target_pois: #타겟 데이터에 대해서 거리 계산
                    avg = 0
                    for poi2 in history_pois:
                        avg += dist(poi1, poi2)
                    avg /= len(history_pois)
                    target_dist.append(avg)
                target_dist=torch.tensor(target_dist,dtype=torch.float32).to(DEVICE)
                prediction = model(user_history, target_list, user_history_region, train_data_region, target_dist)

                _, indices = torch.topk(prediction, args.topk)
                recommended_list.append([target_list[i].item() for i in indices])

                G_score = normalize([G.predict(user_id,poi_id) for poi_id in target_list])
                G_score = torch.tensor(np.array(G_score)).to(DEVICE)
                prediction = (1-alpha)*prediction + alpha * G_score

                _, indices = torch.topk(prediction, args.topk)
                recommended_list_g.append([target_list[i].item() for i in indices])
            end_time = int(time.time())
            print("time: {}".format(end_time-start_time))
            k_list=[5, 10, 15, 20,25,30]
            precision, recall, hit = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
            precision_g, recall_g, hit_g = eval_metrics.evaluate_mp(test_positive,recommended_list_g,k_list)
            

    return recall , precision, hit

def train_BPR(train_matrix, test_positive, test_negative,  dataset):
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
            user, item_i, item_j = get_BPR_batch(train_matrix,test_negative,num_items,buid)
            model.zero_grad() # 그래디언트 초기화
            prediction_i, prediction_j = model(user, item_i, item_j)
            loss = - (prediction_i - prediction_j).sigmoid().log().sum() # BPR 손실 함수 계산
            lo += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        
        elapsed_time = time.time() - start_time  # 소요시간 계산
        print(lo)
        print(f"Epoch : {epoch}, Used_Time : {elapsed_time}")

        if epoch % 20==0:
            model.eval() # 모델을 평가 모드로 설정
            k_list=[5, 10, 15, 20,25,30]
            alpha = args.powerlaw_weight

            recommended_list = []
            recommended_list_g = []
            for user_id in range(num_users):
                user_tensor = torch.LongTensor([user_id] * (len(test_negative[user_id])+len(test_positive[user_id]))).to(DEVICE)
                target_list = test_negative[user_id]+test_positive[user_id]
                target_tensor = torch.LongTensor(target_list).to(DEVICE)

                prediction, _ = model(user_tensor, target_tensor,target_tensor)

                _, indices = torch.topk(prediction, args.topk)
                recommended_list.append([target_list[i] for i in indices])

                G_score = normalize([G.predict(user_id,poi_id) for poi_id in target_list])
                G_score = torch.tensor(np.array(G_score)).to(DEVICE)
                prediction = (1-alpha)*prediction + alpha * G_score

                _, indices = torch.topk(prediction, args.topk)
                recommended_list_g.append([target_list[i] for i in indices])


            precision, recall, hit = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
            precision_g, recall_g, hit_g = eval_metrics.evaluate_mp(test_positive,recommended_list_g,k_list)

def get_bpr_test_input_mp(start_uid,end_uid,test_negative,test_positive):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_tensor_list = []
    target_tensor_list = []
    for user_id in range(start_uid,end_uid):
        user_tensor = torch.LongTensor([user_id] * (len(test_negative[user_id])+len(test_positive[user_id])))
        target_list = test_negative[user_id]+test_positive[user_id]
        target_tensor = torch.LongTensor(target_list)
        user_tensor_list.append(user_tensor)
        target_tensor_list.append(target_tensor)

    return (user_tensor_list,target_tensor_list)
        
def main():
    dataset_ = datasets.Yelp()
    train_matrix,  test_positive, test_negative, place_coords = dataset_.generate_data(0)
    # datasets.get_region(place_coords,200,dataset_.directory_path)
    # datasets.get_region_num(dataset_.directory_path)
    print("data generated")

    G.fit_distance_distribution(train_matrix, place_coords)
    train_BPR(train_matrix, test_positive, test_negative, dataset_)


if __name__ == '__main__':
    G = PowerLaw()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
