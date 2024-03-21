from batches import *
import torch.cuda
import eval_metrics
import numpy as np

def NAIS_validation(model, args,num_users, test_positive, val_positive, train_matrix,k_list):
    model.eval() # 모델을 평가 모드로 설정
    recommended_list = []
    train_loss=0.0
    for user_id in range(num_users):
        user_history, target_list, train_label = get_NAIS_batch_test(train_matrix,user_id)
        prediction = model(user_history, target_list)
        # loss = model.loss_func(prediction,train_label)
        # train_loss += loss.item()
        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
    
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t
    # return 0,[train_loss],0

def NAIS_region_validation(model, args,num_users, test_positive, val_positive, train_matrix, businessRegionEmbedList,k_list):
    model.eval() # 모델을 평가 모드로 설정
    recommended_list = []
    train_loss=0.0
    for user_id in range(num_users):
        user_history, target_list, train_label, user_history_region, train_data_region = get_NAIS_batch_test_region(train_matrix,user_id, businessRegionEmbedList)

        prediction = model(user_history, target_list, user_history_region, train_data_region)
        # loss = model.loss_func(prediction,train_label)
        # train_loss += loss.item()
    
        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i].item() for i in indices])

    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t


def NAIS_region_distance_validation(model, args,num_users, test_positive, val_positive, train_matrix, businessRegionEmbedList, latlon_mat,k_list):
    model.eval() # 모델을 평가 모드로 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    alpha = args.powerlaw_weight
    recommended_list = []
    
    for user_id in range(num_users):
        # torch.cuda.empty_cache()
        # user_history, target_list, train_label, user_history_region, train_data_region = get_NAIS_batch_test_region(train_matrix,user_id, businessRegionEmbedList)         

        # history_pois = user_history[0].tolist() # 방문한 데이터
        # target_pois = target_list.tolist() # 타겟 데이터
            
        # history_pois = np.repeat(np.array(history_pois).reshape(1,-1),len(target_pois),axis=0)
        # target_pois = np.repeat(np.array(target_pois).reshape(-1,1),len(user_history[0]),axis=1)
        # target_lat_long = latlon_mat[target_pois ,history_pois]

        # target_lat_long=torch.tensor(target_lat_long,dtype=torch.float32).to(DEVICE)
        # prediction = model(user_history, target_list, user_history_region, train_data_region, target_lat_long)
        # _, indices = torch.topk(prediction, args.topk)
        # recommended_list.append([target_list[i].item() for i in indices])
        re = []
        ta=[]
        history = train_matrix.getrow(user_id).indices.tolist()
        negatives = list(set(range(train_matrix.shape[1]))-set(history))
        bsize = 2048 
        negative = [negatives[i * bsize:(i + 1) * bsize] for i in range((len(negatives) + bsize - 1) // bsize )]
        for bat in range(len(negative)):
            histories = np.array([history]).repeat(len(negative[bat]),axis=0)

            user_history = histories
            train_data = negative[bat]
            
            user_history_region = businessRegionEmbedList[history]
            
            user_history_region = np.array([user_history_region]).repeat(len(user_history),axis=0)

            train_data_region = businessRegionEmbedList[train_data]

            user_history=torch.LongTensor(user_history).to(DEVICE)
            train_data=torch.LongTensor(train_data).to(DEVICE)
            user_history_region=torch.LongTensor(user_history_region).to(DEVICE)
            train_data_region=torch.LongTensor(train_data_region).to(DEVICE)      
            target_list = train_data

            history_pois = user_history[0].tolist() # 방문한 데이터
            target_pois = target_list.tolist() # 타겟 데이터
            
            history_pois = np.repeat(np.array(history_pois).reshape(1,-1),len(target_pois),axis=0)
            target_pois = np.repeat(np.array(target_pois).reshape(-1,1),len(user_history[0]),axis=1)
            target_lat_long = latlon_mat[target_pois ,history_pois]
            # target_lat_long = []
            # for poi1 in target_pois: #타겟 데이터에 대해서 거리 계산 batch_size
            #     hist = latlon_mat[[poi1] ,history_pois]
            #     target_lat_long.append(hist.tolist())
            target_lat_long=torch.tensor(target_lat_long,dtype=torch.float32).to(DEVICE)
            prediction = model(user_history, target_list, user_history_region, train_data_region, target_lat_long)
            ta.append(target_list)
            re.append(prediction)
        # _, indices = torch.topk(prediction, args.topk)
        
        prediction = torch.cat(re,dim=-1)
        target_list = torch.cat(ta,dim=-1)
        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i].item() for i in indices])

    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t

def BPR_validation(model, args,num_users, test_positive, val_positive, train_matrix, k_list):
    model.eval() # 모델을 평가 모드로 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    recommended_list = []
    for user_id in range(num_users):
        history = train_matrix.getrow(user_id).indices.tolist()
        target_list = list(set(range(train_matrix.shape[1])) - set(history))
        user_tensor = torch.LongTensor([user_id] * (len(target_list))).to(DEVICE)
        target_tensor = torch.LongTensor(target_list).to(DEVICE)

        prediction, _ = model(user_tensor, target_tensor,target_tensor)

        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i] for i in indices])

    
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t

def GPR_validation(model, args,num_users, test_positive, val_positive, train_matrix, k_list):
    model.eval()
    recommended_list = []
    for user_id in range(num_users):
        print(user_id)
        user_id, target_list = get_GPR_batch_test(train_matrix,user_id)
        rating_ul, rating_ul_prime, e_ij_hat = model(user_id, target_list, target_list)
        _, indices = torch.topk(rating_ul.squeeze(), args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
        
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t

def GPR_validation_(model, args,num_users, test_data, test_positive,val_positive,k_list):
    model.eval()
    recommended_list = []
    for user_id in range(num_users):
        print(user_id)
        user_id, target_list = test_data[user_id]
        rating_ul, rating_ul_prime, e_ij_hat = model(user_id, target_list, target_list)
        _, indices = torch.topk(rating_ul.squeeze(), args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
        
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t

def GeoIE_validation(model, args,num_users, test_positive, val_positive, train_matrix,k_list,dist_mat):
    model.eval()
    recommended_list = []
    train_loss=0.0
    for user_id in range(num_users):
        user_id, user_history, target_list, train_label, freq, distances = get_GeoIE_batch_test(train_matrix,user_id, dist_mat)
        prediction, w = model(user_id, target_list, user_history, freq, distances)
        _, indices = torch.topk(prediction.squeeze(), args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
    
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t

def GeoIE_validation_(model, args,num_users,test_data, test_positive, val_positive,k_list):
    model.eval()
    recommended_list = []
    train_loss=0.0
    for uid in range(num_users):
        user_id, user_history, target_list, train_label, freq, distances = test_data[uid]
        prediction, w = model(user_id, target_list, user_history, freq, distances)
        _, indices = torch.topk(prediction.squeeze(), args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
    
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t


def New1_validation(model, args,num_users, test_positive, val_positive, train_matrix, businessRegionEmbedList,k_list):
    with torch.no_grad():
        model.eval() # 모델을 평가 모드로 설정

        recommended_list = []
        train_loss=0.0
        for user_id in range(num_users):
            user_history, target_list, train_label, user_history_region, train_data_region, visit_rate = get_New1_test(train_matrix,user_id, businessRegionEmbedList)

            prediction = model(user_history, target_list, user_history_region, train_data_region, visit_rate)
            # loss = model.loss_func(prediction,train_label)
            # train_loss += loss.item()
        
            _, indices = torch.topk(prediction, args.topk)
            recommended_list.append([target_list[i].item() for i in indices])

        precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
        precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t