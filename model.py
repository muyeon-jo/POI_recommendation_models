import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torchmetrics.functional.pairwise import pairwise_manhattan_distance
from haversine import haversine

class NAIS_basic(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta): # embed_size : 64, beta : 0.5
        super(NAIS_basic, self).__init__()
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_history = nn.Embedding(item_num, self.embed_size) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, self.embed_size) # (m:14586 * d:64), 예측 데이터(p)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy
        self.drop = nn.Dropout()

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(self.embed_size, self.hidden_size)
        self.attn_layer2 = nn.Linear(self.hidden_size, 1, bias = False)

        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)

        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        history_tensor = history
        target_tensor = target
        #print(f"history_tensor : {history_tensor.shape}")
        #print(f"target_tensor : {target_tensor.shape}")

        prediction = self.attention_network(history_tensor,target_tensor)
        # nan 값을 True 로 리턴하는 isnan 메서드 활용
        mask = torch.isnan(prediction)
        mask_int= mask.int()
        nan_count = mask_int.sum().item()
        if nan_count>0:
            print(nan_count)
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item):
        """
        b: batch size
        n: history size
        d: embedding size
        """
        
        history = self.embed_history(user_history) # (b * n * d)

        target = self.embed_target(target_item) # (b * 1 * d)

        batch_dim = len(target)
        target = torch.reshape(target,(batch_dim, 1,-1))
        input = history * target # (b * n * d)
        result1 = self.relu(self.drop(self.attn_layer1(input))) # (n * d)
        
        result2 = self.attn_layer2(result1) # (n * 1) 
        
        exp_A = torch.exp(result2) # (b * n * 1)
        exp_A = exp_A.squeeze(dim=-1)# (b * n )
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)
        result = history * attn_weights# (b * n * d)
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)
        
        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction = torch.sum(prediction, dim = -1) # (b)
        return prediction
        

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)

class NAIS_regionEmbedding(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size): # embed_size : 64, beta : 0.5
        super(NAIS_regionEmbedding, self).__init__()
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_history = nn.Embedding(item_num, int(embed_size/2)) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, int(embed_size/2)) # (m:14586 * d:64), 예측 데이터(p)
        
        self.embed_region = nn.Embedding(region_embed_size, int(embed_size/2)) # 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)
        self.drop = nn.Dropout()
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, history_region, target_region):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        history_tensor = history
        target_tensor = target
        
        history_region_idx = history_region
        target_region_idx = target_region

        prediction = self.attention_network(history_tensor,target_tensor, history_region_idx, target_region_idx)
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region, target_region):
        """
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        
        history_ = self.embed_history(user_history) # (b * n * d)
        region = self.embed_region(history_region) # (b * n * d)
        history = torch.cat((history_, region), -1)

        target_ = self.embed_target(target_item) # (b * 1 * d)
        target_region_ = self.embed_region(target_region) # (b * 1 * d)
        target = torch.cat((target_, target_region_),-1)

        batch_dim = len(target)
        target = torch.reshape(target,(batch_dim, 1,-1))
        input = history * target # (b * n * d)
        result1 = self.relu(self.drop(self.attn_layer1(input))) # (n * d)
        
        result2 = self.attn_layer2(result1) # (n * 1) 
        
        exp_A = torch.exp(result2) # (b * n * 1)
        exp_A = exp_A.squeeze(dim=-1)# (b * n )
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)
        result = history * attn_weights# (b * n * d)
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)
        
        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction = torch.sum(prediction, dim = -1) # (b)
        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)

class NAIS_region_distance_Embedding(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size, dist_embed_size): # embed_size : 64, beta : 0.5, region_embed_size : 152 dist_embed_size : 1
        super(NAIS_region_distance_Embedding, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.DEVICE = 'cpu'
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_history = nn.Embedding(item_num, int(embed_size/2)) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, int(embed_size/2)) # (m:14586 * d:64), 예측 데이터(p)
        # with open(".\data\Yelp\poi_region_sorted.txt", 'r') as file:
        #     # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        #     loaded_embed = [int(line.split('\t')[1].strip()) for line in file.readlines()]
        self.embed_region = nn.Embedding(region_embed_size, int(embed_size/2)) # 
        self.embed_distance = nn.Embedding(dist_embed_size, embed_size) # 
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size + 2, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self.dist_layer = nn.Linear(2,2)

        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        nn.init.normal_(self.embed_distance.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, history_region, target_region, target_lat_long):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        history_tensor = history
        target_tensor = target
        
        
        history_region_idx = history_region
        target_region_idx = target_region
        
        target_lat_long_tensor = target_lat_long

        prediction = self.attention_network(history_tensor,target_tensor, history_region_idx, target_region_idx, target_lat_long_tensor)
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region, target_region, target_lat_long_tensor):
        """
        b: batch size 
        h: history size
        d: embedding size
        k: hidden dim
        """
        history_ = self.embed_history(user_history) # (b * h * d)
        region = self.embed_region(history_region) # (b * h * d)
        history = torch.cat((history_, region), -1) # (b * h * 2d)
        
        target_ = self.embed_target(target_item) # (b * d)
        target_region_ = self.embed_region(target_region) # (b * d)
        target =  torch.cat((target_, target_region_),-1) # (b * 2d)
        
        batch_dim = len(target) #
        history_dim   = len(history[0])
        target = torch.reshape(target,(batch_dim, 1,-1)) # (b * 2d) -> (b * 1 * 2d)
        
        dist = self.sigmoid(self.dist_layer(target_lat_long_tensor*100))
        input = history * target # (b * h * 2d)
        input = torch.cat((input,dist), dim = -1)
        attention_result = self.relu(self.attn_layer1(input)) # (b * h * k)
        attention_result = self.attn_layer2(attention_result) # (b * h * 1)
        # li = np.zeros((batch_dim,history_dim))
        # dist_embedding = self.embed_distance(torch.LongTensor(li).reshape((batch_dim,history_dim)).to(self.DEVICE)) # (b * h * d)
        
        # target_distance = target_distance.unsqueeze(-1) # (b * h * 1)
        # distance = torch.sum(dist_embedding * target_distance, dim = -1) # (b * h * 1)
        # distance = distance.unsqueeze(-1) # b => b * 1 * 1
        
        # exp_input = attention_result + distance # (b * h * 1)
        
        exp_A = torch.exp(attention_result) # (b * h * 1) 
        exp_A = exp_A.squeeze(dim=-1)# (b * h)

        mask = self.get_mask(user_history, target_item) # (b * h)
        exp_A = exp_A * mask # (b * h)
        exp_sum = torch.sum(exp_A,dim=-1) # (b)
        exp_sum = torch.pow(exp_sum, self.beta) # (b)

        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * h)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * h * 1)
        result = history * attn_weights# (b * h * 2d) 
        target = target.reshape([batch_dim,-1,1]) # (b * 2d * 1)

        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * h * 1) -> (b * h)
        prediction = torch.sum(prediction, dim = -1) # (b)
        
        #print(f"#### return prediction {prediction.shape},{prediction}")
        #print("########################### def attention network end ##################################")
        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)

class NAIS_distance_Embedding(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size, dist_embed_size): # embed_size : 64, beta : 0.5, region_embed_size : 152 dist_embed_size : 1
        super(NAIS_distance_Embedding, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.DEVICE = 'cpu'
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_history = nn.Embedding(item_num, embed_size) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, embed_size) # (m:14586 * d:64), 예측 데이터(p)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size+ 2, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self.dist_layer = nn.Linear(2,2)

        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, history_region, target_region, target_distance):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        history_tensor = history
        target_tensor = target
        
        
        # history_region_idx = history_region
        # target_region_idx = target_region
        
        target_distance_tensor = target_distance

        prediction = self.attention_network(history_tensor,target_tensor, target_distance_tensor)
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, target_lat_long_tensor):
        """
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        history = self.embed_history(user_history) # (b * h * d)
        target = self.embed_target(target_item) # (b * d)
        
        batch_dim = len(target) #
        history_dim   = len(history[0])
        target = torch.reshape(target,(batch_dim, 1,-1)) # (b * 1 * d)
        
        # input = history * target # (b * h * d)
        dist = self.sigmoid(self.dist_layer(target_lat_long_tensor*1000))
        input = history * target # (b * h * 2d)
        input = torch.cat((input,dist), dim = -1)

        attention_result = self.relu(self.attn_layer1(input)) # (b * h * k)
        attention_result = self.attn_layer2(attention_result) # (b * h * 1)
        # li = np.zeros((batch_dim,history_dim))
        # dist_embedding = self.embed_distance(torch.LongTensor(li).reshape((batch_dim,history_dim)).to(self.DEVICE)) # (b * h * d)
        # target_distance = target_distance.unsqueeze(-1) # (b * h * 1)
        # distance = torch.sum(dist_embedding * target_distance, dim = -1) # (b * h * 1)
        # distance = distance.unsqueeze(-1) # b => b * 1 * 1
        
        exp_input = attention_result # (b * h * 1)
        
        exp_A = torch.exp(exp_input) # (b * h * 1) 
        exp_A = exp_A.squeeze(dim=-1)# (b * h)

        mask = self.get_mask(user_history, target_item) # (b * h)
        exp_A = exp_A * mask # (b * h)
        exp_sum = torch.sum(exp_A,dim=-1) # (b)
        exp_sum = torch.pow(exp_sum, self.beta) # (b)

        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * h)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * h * 1)
        result = history * attn_weights# (b * h * 2d) 
        target = target.reshape([batch_dim,-1,1]) # (b * 2d * 1)

        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * h * 1) -> (b * h)
        prediction = torch.sum(prediction, dim = -1) # (b)
        
        #print(f"#### return prediction {prediction.shape},{prediction}")
        #print("########################### def attention network end ##################################")
        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)

class NAIS_region_distance_disentangled_Embedding(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size, dist_embed_size): # embed_size : 64, beta : 0.5, region_embed_size : 152 dist_embed_size : 1
        super(NAIS_region_distance_disentangled_Embedding, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.DEVICE = 'cpu'
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_history = nn.Embedding(item_num, embed_size) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, embed_size) # (m:14586 * d:64), 예측 데이터(p)
        # with open(".\data\Yelp\poi_region_sorted.txt", 'r') as file:
        #     # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        #     loaded_embed = [int(line.split('\t')[1].strip()) for line in file.readlines()]
        self.embed_region = nn.Embedding(region_embed_size, embed_size) # 
        self.embed_distance = nn.Embedding(dist_embed_size, embed_size) # 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size , hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self.region_attn_layer1 = nn.Linear(embed_size , hidden_size)
        self.region_attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        nn.init.normal_(self.embed_distance.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, history_region, target_region, target_distance):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        history_tensor = history
        target_tensor = target
        
        
        history_region_idx = history_region
        target_region_idx = target_region
        
        target_distance_tensor = target_distance

        prediction = self.attention_network(history_tensor,target_tensor, history_region_idx, target_region_idx, target_distance_tensor)
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region, target_region, target_distance):
        """
        b: batch size 
        h: history size
        d: embedding size
        k: hidden dim
        """

        history = self.embed_history(user_history) # (b * h * d)
        history_region_ = self.embed_region(history_region) # (b * h * d)
        
        target = self.embed_target(target_item) # (b * d)
        target_region_ = self.embed_region(target_region) # (b * d)
        
        batch_dim = len(target) #
        history_dim   = len(history[0])

        target = torch.reshape(target,(batch_dim, 1,-1)) # (b * 2d) -> (b * 1 * 2d)
        target_region_ = torch.reshape(target_region_,(batch_dim,1,-1))

        input = history * target # (b * h * 2d)
        region_input = history_region_ * target_region_

        attention_result = self.relu(self.attn_layer1(input)) # (b * h * k)
        attention_result = self.attn_layer2(attention_result) # (b * h * 1)

        region_att_result = self.relu(self.region_attn_layer1(region_input))
        region_att_result1 = self.region_attn_layer2(region_att_result)


        li = np.zeros((batch_dim,history_dim))
        dist_embedding = self.embed_distance(torch.LongTensor(li).reshape((batch_dim,history_dim)).to(self.DEVICE)) # (b * h * d)
        target_distance = target_distance.unsqueeze(-1) # (b * h * 1)
        distance = torch.sum(dist_embedding * target_distance, dim = -1) # (b * h * 1)
        distance = distance.unsqueeze(-1) # b => b * 1 * 1
        
        exp_input = attention_result + distance # (b * h * 1)
        region_exp_input = region_att_result1 + distance

        exp_A = torch.exp(exp_input) # (b * h * 1) 
        exp_A = exp_A.squeeze(dim=-1)# (b * h)
        exp_region = torch.exp(region_exp_input)
        exp_region = exp_region.squeeze(dim=-1)

        mask = self.get_mask(user_history, target_item) # (b * h)
        exp_A = exp_A * mask # (b * h)
        region_exp = exp_region * mask

        exp_sum = torch.sum(exp_A,dim=-1) # (b)
        exp_sum = torch.pow(exp_sum, self.beta) # (b)
        region_exp_sum = torch.sum(region_exp,dim=-1)
        region_exp_sum = torch.pow(region_exp_sum,self.beta)

        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * h)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * h * 1)
        region_attn_weights = torch.divide(region_exp.T,region_exp_sum).T # (b * h)
        region_attn_weights = region_attn_weights.reshape([batch_dim,-1, 1])# (b * h * 1)

        result = history * attn_weights# (b * h * 2d) 
        region_result = history_region_ * region_attn_weights
        result = torch.cat((result,region_result),dim = -1)

        target = torch.cat((target,target_region_), dim= -1)
        target = target.reshape([batch_dim,-1,1]) # (b * 2d * 1)

        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * h * 1) -> (b * h)
        prediction = torch.sum(prediction, dim = -1) # (b)
        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)

class BPRData(torch.utils.data.Dataset):
    def __init__(self, matrix,
                num_item, num_ng=0, is_training=None, ng_test = None):
        super(BPRData, self).__init__()
        """
            Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        # self.features = features #training, test dataset
        self.matrix = matrix
        self.num_item = num_item
        self.num_ng = num_ng
        self.is_training = is_training
        self.ng_dataset = ng_test

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        #훈련 데이터셋에 대하여 Negativesample 만듦
        self.features_fill = []
        item_list = np.arange(self.num_item).tolist()
        for uid in range(self.matrix.shape[0]):
            idic = self.matrix.getrow(uid).indices
            ng_data_list = list( set(item_list) - set(self.ng_dataset[uid]) - set(list(idic)) )
            for x in idic: # training_dataset에 있는 positive data에 랜덤한 negative data를 생성해 매치해줌
                u, i = uid, x
                for t in range(self.num_ng):
                    j = np.random.randint(0, len(ng_data_list)) # num_items == len(item_list), item_list에서 Negative sample을 골라야하기 때문에 인덱스를 골라줌
                    j = int(ng_data_list[j])
                    self.features_fill.append([u, i, j])


    def __len__(self):
        return self.num_ng * self.matrix.shape[0]

    def __getitem__(self, idx):
        features = self.features_fill 

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
                self.is_training else features[idx][1]

        return user, item_i, item_j

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        """
        user_num: 사용자 수
        item_num: 아이템 수
        factor_num: 예측할 factors의 수
        """
            # 사용자와 아이템을 factor 수 만큼 임베딩하는 레이어를 생성
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        # 임베딩 레이어의 가중치를 정규 분포에서 초기화한다.
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        """
        Parameters:
            user (torch.Tensor): 사용자 ID list
            item_i (torch.Tensor): positive 아이템 list
            item_j (torch.Tensor): negative 아이템 list
        Returns:
            torch.Tensor: 긍정적인 아이템에 대한 예측 값
            torch.Tensor: 부정적인 아이템에 대한 예측 값
        """
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        
        prediction_i = (user * item_i).sum(dim=-1)  # user * i matrix 생성
        prediction_j = (user * item_j).sum(dim=-1) # user * j matrix 생성

        return prediction_i, prediction_j

class GGLR(nn.Module):
    def __init__(self, embed_dim, layer_num):
        super(GGLR, self).__init__()
        self.embed_dim = embed_dim
        self.k = layer_num
        self.a = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.b = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.c = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.ingoing_conv1 = GCNConv(embed_dim,embed_dim)
        self.ingoing_conv2 = GCNConv(embed_dim,embed_dim)

        self.outgoing_conv1 = GCNConv(embed_dim,embed_dim)
        self.outgoing_conv2 = GCNConv(embed_dim,embed_dim)

        self.decode_layer = nn.Linear(embed_dim, embed_dim, bias=False)

        self.mse_loss = nn.MSELoss()
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, p_outgoing, q_ingoing, adjacency_matrix, distance_matrix):
        # adj_mat = adjacency_matrix

        no = adjacency_matrix.clone()
        no[adjacency_matrix > 0.0] = 1
        D_outgoing = torch.sum(no, dim=-1) + 0.0000001
        D_ingoing = torch.sum(no.transpose(0, 1), dim=-1)+ 0.0000001
        
        e_ij_hat = []
        # t =adjacency_matrix.nonzero().T
        outgoing1 = self.outgoing_conv1(p_outgoing, adjacency_matrix.nonzero().T)
        outgoing1 = torch.mm(adjacency_matrix, outgoing1) #(poi * poi) (dot) (poi * emb)
        # tt = D_outgoing.reshape(-1,1)
        # ttt = D_ingoing.reshape(-1,1)
        outgoing1 = torch.div(outgoing1, D_outgoing.reshape(-1,1))
        outgoing1 = self.leaky_relu(outgoing1)

        outgoing2 = self.outgoing_conv2(outgoing1, adjacency_matrix.nonzero().T)
        outgoing2 = torch.mm(adjacency_matrix,outgoing2)
        outgoing2 = torch.div(outgoing2, D_outgoing.reshape(-1,1))
        outgoing2 = self.leaky_relu(outgoing2)
        
        ingoing1 = self.ingoing_conv1(q_ingoing, adjacency_matrix.T.nonzero().T)
        ingoing1 = torch.mm(adjacency_matrix.T, ingoing1) #(poi * poi) (dot) (poi * emb)
        ingoing1 = torch.div(ingoing1,  D_ingoing.reshape(-1,1))
        ingoing1 = self.leaky_relu(ingoing1)

        ingoing2 = self.ingoing_conv2(ingoing1, adjacency_matrix.T.nonzero().T)
        ingoing2 = torch.mm(adjacency_matrix.T,ingoing2)
        ingoing2 = torch.div(ingoing2, D_ingoing.reshape(-1,1))
        ingoing2 = self.leaky_relu(ingoing2)

        fx_ij = torch.mul(torch.mul(distance_matrix**self.b,self.a), torch.exp(torch.mul(distance_matrix,self.c)))
        e_ij_hat = torch.mul(torch.mm(self.decode_layer(outgoing2), ingoing2.T), fx_ij) 
        return [outgoing1,outgoing2], [ingoing1,ingoing2], e_ij_hat
    
    def loss_function(self, ground, predict):
        ground = ground.reshape(-1,1)
        predict = predict.reshape(-1,1)
        return self.mse_loss(ground, predict)
class GPR(nn.Module):
    def __init__(self, user_num, poi_num, embed_dim, layer_num, POI_POI_Graph, distance_matrix,user_POI_Graph, lambda1=0.2):
        super(GPR, self).__init__()
        self.POI_POI_Graph = torch.tensor(POI_POI_Graph, dtype=torch.float32).to("cuda")
        self.distance_matrix = distance_matrix
        self.user_POI_Graph = torch.tensor(user_POI_Graph, dtype= torch.float32).to("cuda")
        self.embed_dim = embed_dim
        self.k = layer_num
        self.user_num = user_num
        self.poi_num = poi_num
        self.sigmoid = nn.Sigmoid()
        self.lambda1 = lambda1
        self.gglr = GGLR(embed_dim, layer_num).to("cuda")

        self.user_embed = nn.Embedding(user_num,embed_dim) # t
        self.p_outgoing_embed = nn.Embedding(poi_num,embed_dim) # t
        self.q_incoming_embed = nn.Embedding(poi_num,embed_dim) # t

       
        
        self.user_layer1 = nn.Linear(embed_dim,embed_dim,bias=False)
        self.user_layer2 = nn.Linear(embed_dim,embed_dim,bias=False)

        self.outgoing_layer1 = GCNConv(embed_dim,embed_dim)
        self.outgoing_layer2 = GCNConv(embed_dim,embed_dim)

        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.p_outgoing_embed.weight)
        nn.init.xavier_normal_(self.q_incoming_embed.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self,user_ids, train_positives, train_negatives ):
        #gglr result
        
        p1 = self.p_outgoing_embed(torch.LongTensor(range(self.poi_num)).to("cuda"))
        q1 = self.q_incoming_embed(torch.LongTensor(range(self.poi_num)).to("cuda"))
        u1 = self.user_embed(torch.LongTensor(range(self.user_num)).to("cuda"))

        p_k, q_k, e_ij_hat = self.gglr(p1, q1, self.POI_POI_Graph, self.distance_matrix)
        temp_user_emb = torch.tensor(np.zeros([self.user_num,self.embed_dim]), dtype= torch.float32, requires_grad = False).to("cuda")
        p_k[0] = torch.cat([p_k[0],temp_user_emb],dim = 0)
        p_k[1] = torch.cat([p_k[1],temp_user_emb],dim = 0)

        user1 = self.user_layer1(u1)
        edge_list = torch.stack([self.user_POI_Graph.nonzero().T[0].add(self.poi_num),self.user_POI_Graph.nonzero().T[1]])
        # p1 = torch.sum(self.outgoing_layer1(p_k[0],edge_list),dim=0)
        user1 = self.sigmoid(user1 + torch.sum(self.outgoing_layer1(p_k[0],edge_list),dim=0))

        user2 = self.user_layer2(user1)
        # p2 = torch.sum(self.outgoing_layer2(p_k[1],edge_list),dim=0)
        user2 = self.sigmoid(user2 + torch.sum(self.outgoing_layer2(p_k[1],edge_list),dim=0))


        result_u = torch.cat((user1,user2), dim=-1)
        result_q = torch.cat((q_k[0],q_k[1]),dim=-1)
        
        tt = result_u[user_ids]
        pp = result_q[train_positives]
        qq = result_q[train_negatives]

        rating_ul = torch.mm(tt, pp.T).diag()
        rating_ul_prime = torch.mm(tt,qq.T).diag()
        return rating_ul, rating_ul_prime, e_ij_hat 
    
    def loss_function(self, rating_ul, rating_ul_p, e_ij_hat):
        loss1 = self.gglr.loss_function(self.POI_POI_Graph,e_ij_hat)
        loss2 = -torch.sum(torch.log(self.sigmoid(rating_ul - rating_ul_p)+ 0.0000001))
        loss = loss2 + loss1*self.lambda1
        return loss


class GeoIE(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,neg_num,a,b):
        super(GeoIE,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.negnum=neg_num
        self.a=a
        self.b=b
        # self.a = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))  # Initialize a
        # self.b = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))  # Initialize b

        
        self.UserPreference=nn.Embedding(user_count,emb_dimension) # t
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension) # z
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension) # g
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension) # h
        self.init_emb()

        self.POI_count = POI_count
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

    def init_emb(self):
        nn.init.xavier_normal_(self.UserPreference.weight)
        nn.init.xavier_normal_(self.PoiPreference.weight)
        nn.init.xavier_normal_(self.GeoInfluence.weight)
        nn.init.xavier_normal_(self.GeoSusceptibility.weight)

    def forward(self, user_id, targets, history, check_in_num, distances):
        # user_id = torch.LongTensor(user_id)  #
        # targets = torch.LongTensor(targets) # 
        # history = torch.LongTensor(history) # 
        cuj = check_in_num # len : target

        UPre = self.UserPreference(user_id) #b * emb 
        PPre = self.PoiPreference(targets) #b * emb 
        hj = self.GeoSusceptibility(targets) #b * emb 

        g = self.GeoInfluence(history) #b * h * emb 
        history_size = len(history[0])
        batch_size = len(history)
        g = g.reshape([batch_size,-1,history_size]) # b * emb * h
        fij = self.a * (distances**self.b)
        hj = torch.reshape(hj,[batch_size,-1,1]) #b * emb * 1
        t1 = g*hj  # b * emb * h
        t2 = torch.sum(t1,dim=1) * fij # b * h
        yij = torch.sum(t2,dim=-1) / history_size 

        UPre = torch.reshape(UPre,[batch_size,1,-1])
        PPre = torch.reshape(PPre,[batch_size,-1,1])

        tz = torch.bmm(UPre,PPre).squeeze(-1) # t * 1
        
        suj = tz + yij.unsqueeze(1) # t * 1

        wuj = 1 + torch.log(1+cuj*(10**self.scaling))
        return self.sigmoid(suj), wuj

    
    def loss_function(self, prediction, label, weight):
        t1 = label * (torch.log(prediction+1e-10))
        t2 = (1-label)*(torch.log(1-prediction+1e-10))
        loss = -weight * (t1+t2)
        loss = torch.sum(loss)
        temp = torch.isnan(loss)
        if temp.item() == True:
            print(prediction)
            print(label)
            print(weight)
            print(t1)
            print(t2)
        return torch.sum(loss)

class New1(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size): # embed_size : 64, beta : 0.5
        super(New1, self).__init__()
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_target = nn.Embedding(item_num, int(embed_size/2)) 
        
        self.embed_region = nn.Embedding(region_embed_size, int(embed_size/2)) # 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_Q = nn.Linear(embed_size, hidden_size, bias = False)
        self.attn_K = nn.Linear(embed_size, hidden_size, bias = False)
        self.attn_V = nn.Linear(embed_size, embed_size, bias = False)
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, history_region, target_region, history_visit_rate):
        history_tensor = history.squeeze(0)
        target_tensor = target.squeeze(0)
        
        history_region_idx = history_region.squeeze(0)
        target_region_idx = target_region.squeeze(0)

        history_visit_rate = history_visit_rate.squeeze(0)
        prediction = self.attention_network(history_tensor,target_tensor, history_region_idx, target_region_idx,history_visit_rate.type(torch.float32))
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region, target_region, history_visit_rate):
        """
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        
        history_ = self.embed_target(user_history) # (b * n * d)
        region = self.embed_region(history_region) # (b * n * d)
        history = torch.cat((history_, region), -1)

        target_ = self.embed_target(target_item) # (b * 1 * d)
        target_region_ = self.embed_region(target_region) # (b * 1 * d)
        target = torch.cat((target_, target_region_),-1)

        batch_dim = len(target)
        embed_dim = target.shape[1]
        target = torch.reshape(target,(batch_dim, 1,-1))

        query_emb = self.attn_Q(target)
        key_emb = self.attn_K(history)
        value_emb = self.attn_V(history)
        
        
        result1 = torch.bmm(query_emb,torch.reshape(key_emb,(batch_dim,embed_dim,-1))) # (n * 1) 
        result2 = result1/torch.sqrt(torch.tensor(embed_dim))
        result2 = result2.squeeze()
        exp_A = torch.exp(result2) # (b * n)
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)
        result1 = value_emb * attn_weights# (b * n * d)
        r = history_visit_rate.repeat(batch_dim,1).reshape(batch_dim,-1,1)
        result2 = history * r# (b * n * d)
        result = result1 + result2 # (b * n * d)
        
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)
        
        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction = torch.sum(prediction, dim = -1) # (b)


        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)

class New2(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size, user_num): # embed_size : 64, beta : 0.5
        super(New2, self).__init__()
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.user_num = user_num
        self.embed_target = nn.Embedding(item_num, int(embed_size/2)) 
        
        self.embed_region = nn.Embedding(region_embed_size, int(embed_size/2)) # 
        self.embed_dist = nn.Parameter(torch.normal(0.0,0.01,(user_num,region_embed_size)))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_Q = nn.Linear(embed_size, hidden_size, bias = False)
        self.attn_K = nn.Linear(embed_size, hidden_size, bias = False)
        self.attn_V = nn.Linear(embed_size, embed_size, bias = False)
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, history_region, target_region, history_visit_rate, dist_mat,uid):
        history_tensor = history.squeeze(0)
        target_tensor = target.squeeze(0)
        
        history_region_idx = history_region.squeeze(0)
        target_region_idx = target_region.squeeze(0)

        history_visit_rate = history_visit_rate.squeeze(0)
        prediction = self.attention_network(history_tensor,target_tensor, history_region_idx, target_region_idx,history_visit_rate.type(torch.float32),dist_mat.type(torch.float32),uid)
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region, target_region, history_visit_rate,dist,uid):
        """
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        
        history_ = self.embed_target(user_history) # (b * n * d)
        region = self.embed_region(history_region) # (b * n * d)
        history_dist_emb = self.embed_dist[uid][history_region]
        history = torch.cat((history_, region), -1)

        target_ = self.embed_target(target_item) # (b * 1 * d)
        target_region_ = self.embed_region(target_region) # (b * 1 * d)
        target_dist_emb = self.embed_dist[uid][target_region]
        target = torch.cat((target_, target_region_),-1)

        batch_dim = len(target)
        embed_dim = target.shape[1]
        target = torch.reshape(target,(batch_dim, 1,-1))

        query_emb = self.attn_Q(target)
        key_emb = self.attn_K(history)
        value_emb = self.attn_V(history)
        
        
        result1 = torch.bmm(query_emb,torch.reshape(key_emb,(batch_dim,embed_dim,-1))) # (n * 1) 
        result2 = result1/torch.sqrt(torch.tensor(embed_dim))
        result2 = result2.squeeze()
        exp_A = torch.exp(result2) # (b * n)
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)

        geo_weights = torch.exp(-1/(self.relu(target_dist_emb @ history_dist_emb)+1.0)*dist.reshape(batch_dim,-1)).reshape([batch_dim,-1, 1])
        result1 = value_emb * (attn_weights + geo_weights)# (b * n * d)
        r = history_visit_rate.repeat(batch_dim,1).reshape(batch_dim,-1,1)
        result2 = history * r# (b * n * d)
        result = result1 + result2 # (b * n * d)
        
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)
        
        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction = torch.sum(prediction, dim = -1) # (b)


        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)
    


class New3(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(New3, self).__init__()
        """
        user_num: 사용자 수
        item_num: 아이템 수
        factor_num: 예측할 factors의 수
        """
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = factor_num
        self.beta = 0.5

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        self.embed_ingoing = nn.Embedding(item_num,factor_num)
        self.embed_outgoing = nn.Embedding(item_num,factor_num)

        self.query = nn.Linear(factor_num*2,factor_num*2)
        self.key = nn.Linear(factor_num*2,factor_num*2)
        self.value = nn.Linear(factor_num*2,factor_num*2)

        self.attn_Q = nn.Linear(factor_num*3,factor_num*3)
        self.attn_K = nn.Linear(factor_num*3,factor_num*3)
        self.attn_V = nn.Linear(factor_num*3,factor_num*3)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_ingoing.weight, std=0.01)
        nn.init.normal_(self.embed_outgoing.weight, std=0.01)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss_func = nn.BCELoss()

    def forward(self, user, item_i, item_j):
        """
        """
        
        region_embed = self.self_attention(self.embed_ingoing(torch.arange(0,self.item_num).cuda()),self.embed_outgoing(torch.arange(0,self.item_num).cuda()))

        pos_region = region_embed[item_i]
        neg_region = region_embed[item_j]

        return self.attention_network(user,item_i, item_j,region_embed[user],pos_region,neg_region)

        # prediction_i = (user * pos).sum(dim=-1)
        # prediction_j = (user * neg).sum(dim=-1)

        # return prediction_i, prediction_j
    def self_attention(self, ingoing, outgoing):
        # q = self.query(torch.cat((ingoing,outgoing),dim=-1))
        # k = self.key(torch.cat((outgoing,ingoing),dim=-1))
        # v = self.value(torch.cat((ingoing,outgoing),dim=-1))

        q = torch.cat((ingoing,outgoing),dim=-1)
        k = torch.cat((outgoing,ingoing),dim=-1)
        v = torch.cat((ingoing,outgoing),dim=-1)
        t3 = self.softmax(q @ k.T/torch.sqrt(torch.tensor(self.embed_dim*2)))
        
        t4 = t3@v
        return t4
    def attention_network(self, user_history, pos_item, neg_item, history_region, pos_region, neg_region):
        history_ = self.embed_item(user_history) # (b * n * d)
        history = torch.cat((history_, history_region), -1)

        pos = self.embed_item(pos_item) # (b * 1 * d)
        neg = self.embed_item(neg_item) # (b * 1 * d)
        pos_target = torch.cat((pos, pos_region),-1)
        neg_target = torch.cat((neg, neg_region),-1)

        batch_dim = len(pos_target)
        embed_dim = pos_target.shape[1]
        pos_target = torch.reshape(pos_target,(batch_dim, 1,-1))
        neg_target = torch.reshape(neg_target,(batch_dim, 1,-1))

        query_pos_emb = self.attn_Q(pos_target)
        query_neg_emb = self.attn_Q(neg_target)
        key_emb = self.attn_K(history)
        value_emb = self.attn_V(history)
        
        
        result_pos = torch.bmm(query_pos_emb,torch.reshape(key_emb,(batch_dim,embed_dim,-1))) # (n * 1) 
        result_neg = torch.bmm(query_neg_emb,torch.reshape(key_emb,(batch_dim,embed_dim,-1))) # (n * 1) 
        result_pos = result_pos/torch.sqrt(torch.tensor(embed_dim))
        result_neg = result_neg/torch.sqrt(torch.tensor(embed_dim))
        result_pos = result_pos.squeeze()
        result_neg = result_neg.squeeze()
        
        exp_A = torch.exp(result_pos) # (b * n)
        mask = self.get_mask(user_history,pos_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)

        result_pos = value_emb * attn_weights# (b * n * d)
        
        # 
        exp_A = torch.exp(result_neg) # (b * n)
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)

        result_neg = value_emb * attn_weights# (b * n * d)
        
        prediction_i = torch.bmm(result_pos, pos_target.reshape([batch_dim,-1,1])).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction_i = torch.sum(prediction_i, dim = -1) # (b)

        prediction_j = torch.bmm(result_neg, neg_target.reshape([batch_dim,-1,1])).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction_j = torch.sum(prediction_j, dim = -1) # (b)
        return prediction_i, prediction_j

    def bpr_loss(self,prediction_i,prediction_j):
        return - self.sigmoid(prediction_i - prediction_j).log().sum()
    def loss_function(self, prediction_i, prediction_j):
        prediction = self.sigmoid(torch.cat((prediction_i,prediction_j),dim=-1))
        label = torch.cat((torch.ones(len(prediction_i)), torch.zeros(len(prediction_j))),dim=-1).cuda()
        return self.loss_func(prediction, label)
    def topk_intersection(self):
        ingoing = self.embed_ingoing(torch.arange(0,self.item_num).cuda())
        outgoing = self.embed_outgoing(torch.arange(0,self.item_num).cuda())
        i_ingoing = ingoing @ outgoing.T
        i_outgoing = outgoing @ ingoing.T
        top_ingoing = torch.topk(i_ingoing,10,dim=-1)
        top_outgoing = torch.topk(i_outgoing,10,dim=-1)
        return top_ingoing, top_outgoing
    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    

class New4(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size): # embed_size : 64, beta : 0.5
        super(New4, self).__init__()
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_ingoing = nn.Embedding(item_num,int(embed_size/4))
        self.embed_outgoing = nn.Embedding(item_num,int(embed_size/4))

        self.embed_history = nn.Embedding(item_num, int(embed_size/2)) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, int(embed_size/2)) # (m:14586 * d:64), 예측 데이터(p)
        
        self.embed_region = nn.Embedding(region_embed_size, int(embed_size/2)) # 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy
        self.softmax = nn.Softmax(dim=-1)

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self.query = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.key = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.value = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.drop = nn.Dropout()
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        nn.init.normal_(self.embed_ingoing.weight, std=0.01)
        nn.init.normal_(self.embed_outgoing.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, near_pois, target_region):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        region_in, region_out = self.self_attention(self.embed_ingoing(torch.LongTensor(near_pois).cuda()),self.embed_outgoing(torch.LongTensor(near_pois).cuda()))
        history_tensor = history
        target_tensor = target
        
        # history_region_idx = history_region
        # target_region_idx = target_region

        prediction = self.attention_network(history_tensor,target_tensor, torch.cat((region_in[history],region_out[history]),dim=-1), torch.cat((region_out[target],region_in[target]),dim=-1))
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region_embed, target_region_embed):
        """
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        
        history_ = self.embed_history(user_history) # (b * n * d)
        history = torch.cat((history_, history_region_embed), -1)

        target_ = self.embed_target(target_item) # (b * 1 * d)
        target = torch.cat((target_, target_region_embed),-1)

        batch_dim = len(target)
        target = torch.reshape(target,(batch_dim, 1,-1))
        input = history * target # (b * n * d)
        input = self.drop(self.attn_layer1(input))
        result1 = self.relu(input) # (n * d)
        
        result2 = self.attn_layer2(result1) # (n * 1) 
        
        exp_A = torch.exp(result2) # (b * n * 1)
        exp_A = exp_A.squeeze(dim=-1)# (b * n )
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)
        result = history * attn_weights# (b * n * d)
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)
        
        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction = torch.sum(prediction, dim = -1) # (b)
        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)
    
    def self_attention(self, ingoing, outgoing):
        # q = self.query(torch.cat((ingoing,outgoing),dim=-1))
        # k = self.key(torch.cat((outgoing,ingoing),dim=-1))
        # v = self.value(torch.cat((ingoing,outgoing),dim=-1))

        # q = torch.cat((ingoing[:,0,:],outgoing[:,0,:]),dim=-1).reshape(self.item_num,1,int(self.embed_size/2))
        # k = torch.cat((outgoing,ingoing),dim=-1).reshape(self.item_num,int(self.embed_size/2),-1)
        # v = torch.cat((ingoing,outgoing),dim=-1)
        
        # t3 = self.softmax(torch.bmm(q,k)/torch.sqrt(torch.tensor(self.embed_size/2)))
        # t4 = torch.bmm(t3,v).squeeze()

        q = ingoing[:,0,:].reshape(self.item_num,1,int(self.embed_size/4))
        k_out = outgoing.reshape(self.item_num,int(self.embed_size/4),-1)
        v_out = outgoing
        
        t3 = self.softmax(torch.bmm(q,k_out)/torch.sqrt(torch.tensor(self.embed_size/4)))
        result_out = torch.bmm(t3,v_out).squeeze()

        q = outgoing[:,0,:].reshape(self.item_num,1,int(self.embed_size/4))
        k_in = ingoing.reshape(self.item_num,int(self.embed_size/4),-1)
        v_in = ingoing
        
        t3 = self.softmax(torch.bmm(q,k_in)/torch.sqrt(torch.tensor(self.embed_size/4)))
        result_in = torch.bmm(t3,v_in).squeeze()
        return result_in,result_out
    

    def topk_intersection(self):
        ingoing = self.embed_ingoing(torch.arange(0,self.item_num).cuda())
        outgoing = self.embed_outgoing(torch.arange(0,self.item_num).cuda())
        i_ingoing = ingoing @ outgoing.T
        i_outgoing = outgoing @ ingoing.T
        top_ingoing = torch.topk(i_ingoing,10,dim=-1)
        top_outgoing = torch.topk(i_outgoing,10,dim=-1)
        return top_ingoing, top_outgoing
    


class New4_padding(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size): # embed_size : 64, beta : 0.5
        super(New4_padding, self).__init__()
        self.embed_size = embed_size # concat 연산 시 * 2
        self.item_num = item_num
        self.beta = beta
        self.hidden_size=hidden_size
        self.embed_ingoing = nn.Embedding(item_num+1,int(embed_size/4),padding_idx=0)
        self.embed_outgoing = nn.Embedding(item_num+1,int(embed_size/4),padding_idx=0)

        self.embed_history = nn.Embedding(item_num+1, int(embed_size/2),padding_idx=0) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num+1, int(embed_size/2),padding_idx=0) # (m:14586 * d:64), 예측 데이터(p)
        
        self.embed_region = nn.Embedding(region_embed_size+1, int(embed_size/2),padding_idx=0) # 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy
        self.softmax = nn.Softmax(dim=-1)

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self.query = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.key = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.value = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.drop = nn.Dropout()
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화, 표준편차 : 0.01
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_region.weight, std=0.01)
        nn.init.normal_(self.embed_ingoing.weight, std=0.01)
        nn.init.normal_(self.embed_outgoing.weight, std=0.01)
        
        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, near_pois, target_region):
        #배치 사이즈만큼 잘라서 넣어줌
        #print(len(history), len(target))
        region_in, region_out = self.self_attention(self.embed_ingoing(torch.LongTensor(near_pois).cuda()),self.embed_outgoing(torch.LongTensor(near_pois).cuda()))
        history_tensor = history
        target_tensor = target
        
        # history_region_idx = history_region
        # target_region_idx = target_region

        prediction = self.attention_network(history_tensor,target_tensor, torch.cat((region_in[history],region_out[history]),dim=-1), torch.cat((region_out[target],region_in[target]),dim=-1))
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, history_region_embed, target_region_embed):
        """
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        
        history_ = self.embed_history(user_history) # (b * n * d)
        history = torch.cat((history_, history_region_embed), -1)

        target_ = self.embed_target(target_item) # (b * 1 * d)
        target = torch.cat((target_, target_region_embed),-1)

        batch_dim = len(target)
        target = torch.reshape(target,(batch_dim, 1,-1))
        input = history * target # (b * n * d)
        input = self.drop(self.attn_layer1(input))
        result1 = self.relu(input) # (n * d)
        
        result2 = self.attn_layer2(result1) # (n * 1) 
        
        exp_A = torch.exp(result2) # (b * n * 1)
        exp_A = exp_A.squeeze(dim=-1)# (b * n )
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) # (b * 1)
        exp_sum = torch.pow(exp_sum, self.beta) # (b * 1)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * n)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * n * 1)
        result = history * attn_weights# (b * n * d)
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)
        
        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * n * 1) -> (b * n)
        prediction = torch.sum(prediction, dim = -1) # (b)
        return prediction

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)
    
    def self_attention(self, ingoing, outgoing):
        # q = self.query(torch.cat((ingoing,outgoing),dim=-1))
        # k = self.key(torch.cat((outgoing,ingoing),dim=-1))
        # v = self.value(torch.cat((ingoing,outgoing),dim=-1))

        # q = torch.cat((ingoing[:,0,:],outgoing[:,0,:]),dim=-1).reshape(self.item_num,1,int(self.embed_size/2))
        # k = torch.cat((outgoing,ingoing),dim=-1).reshape(self.item_num,int(self.embed_size/2),-1)
        # v = torch.cat((ingoing,outgoing),dim=-1)
        
        # t3 = self.softmax(torch.bmm(q,k)/torch.sqrt(torch.tensor(self.embed_size/2)))
        # t4 = torch.bmm(t3,v).squeeze()

        q = ingoing[:,0,:].reshape(self.item_num,1,int(self.embed_size/4))
        k_out = outgoing.reshape(self.item_num,int(self.embed_size/4),-1)
        v_out = outgoing
        
        t3 = self.softmax(torch.bmm(q,k_out)/torch.sqrt(torch.tensor(self.embed_size/4)))
        result_out = torch.bmm(t3,v_out).squeeze()

        q = outgoing[:,0,:].reshape(self.item_num,1,int(self.embed_size/4))
        k_in = ingoing.reshape(self.item_num,int(self.embed_size/4),-1)
        v_in = ingoing
        
        t3 = self.softmax(torch.bmm(q,k_in)/torch.sqrt(torch.tensor(self.embed_size/4)))
        result_in = torch.bmm(t3,v_in).squeeze()
        return result_in,result_out
    

    def topk_intersection(self):
        ingoing = self.embed_ingoing(torch.arange(0,self.item_num).cuda())
        outgoing = self.embed_outgoing(torch.arange(0,self.item_num).cuda())
        i_ingoing = ingoing @ outgoing.T
        i_outgoing = outgoing @ ingoing.T
        top_ingoing = torch.topk(i_ingoing,10,dim=-1)
        top_outgoing = torch.topk(i_outgoing,10,dim=-1)
        return top_ingoing, top_outgoing