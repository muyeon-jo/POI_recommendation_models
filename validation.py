from batches import get_NAIS_batch_test_region,get_NAIS_batch_region,get_NAIS_batch_test,get_NAIS_batch,get_BPR_batch
import torch
import eval_metrics
from powerLaw import dist
def NAIS_validation(model, args,num_users, positive, negative, train_matrix,val_flag,k_list):
    model.eval() # 모델을 평가 모드로 설정
    recommended_list = []
    for user_id in range(num_users):
        user_history, target_list, train_label = get_NAIS_batch_test(train_matrix,positive,negative,user_id)
        prediction = model(user_history, target_list)

        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i].item() for i in indices])
    
    precision, recall, hit = eval_metrics.evaluate_mp(positive,recommended_list,k_list,val_flag)
    
    return precision, recall, hit

def NAIS_region_validation(model, args,num_users, positive, negative, train_matrix, businessRegionEmbedList, val_flag,k_list):
    model.eval() # 모델을 평가 모드로 설정
    recommended_list = []
    for user_id in range(num_users):
        user_history, target_list, train_label, user_history_region, train_data_region = get_NAIS_batch_test_region(train_matrix,positive,negative,user_id, businessRegionEmbedList)

        prediction = model(user_history, target_list, user_history_region, train_data_region)

        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i].item() for i in indices])

    precision, recall, hit = eval_metrics.evaluate_mp(positive,recommended_list,k_list,val_flag)

    return precision, recall, hit

def NAIS_region_distance_validation(model, args,num_users, positive, negative, train_matrix, businessRegionEmbedList, poi_coos,val_flag,k_list):
    model.eval() # 모델을 평가 모드로 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = args.powerlaw_weight
    recommended_list = []
    for user_id in range(num_users):
        user_history, target_list, train_label, user_history_region, train_data_region = get_NAIS_batch_test_region(train_matrix,positive,negative,user_id, businessRegionEmbedList)        
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

    precision, recall, hit = eval_metrics.evaluate_mp(positive,recommended_list,k_list,val_flag)
            

    return recall , precision, hit