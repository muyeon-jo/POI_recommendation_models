import torch
import torch.nn as nn
import numpy as np

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
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        #print("#### def attention network start #####")
        #print(len(user_history), user_history)
        #print(len(target_item), target_item)

        history = self.embed_history(user_history) # (b * d)
        target = self.embed_target(target_item) # (b * d)
        batch_dim = len(target) # (b)

        target = torch.reshape(target,(batch_dim, 1,-1))

        input = history * target # (b*d) * (b*1*d) => (b * b * d)

        attention_result = self.relu(self.attn_layer1(input)) # (b * b * d)

        attention_result = self.attn_layer2(attention_result) # (b * b * 1)

        exp_A = torch.exp(attention_result) # (b * b * 1)
        #print(f"exp_A : {exp_A.shape}")
        exp_A = exp_A.squeeze(dim=-1)# (b * b)
        #print(f"exp_A_Squeeze : {exp_A.shape}")

        mask = self.get_mask(user_history,target_item) # (b * b)
        #print(f"mask : {mask.shape}")
        exp_A = exp_A * mask # (b * b)
        #print(f"exp_A_mask : {exp_A.shape}")
        exp_sum = torch.sum(exp_A,dim=-1) # (b)
        #print(f"exp_A_sum : {exp_sum.shape}")
        exp_sum = torch.pow(exp_sum, self.beta) # (b)
        #print(f"exp_A_pow : {exp_sum.shape}")

        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * b)
        #print(f"attn_weights = {attn_weights.shape}")
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * b * 1)
        #print(f"attn_weights_reshaped = {attn_weights.shape}")
        result = history * attn_weights# (b * b * d)
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)


        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * b * 1) -> (b * b)
        prediction = torch.sum(prediction, dim = -1) # (b)
        #print(f"#### return prediction{prediction.shape}")
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
        self.embed_history = nn.Embedding(item_num, embed_size) # (m:14586 * d:64), 과거 방문한 데이터(q), 유저별로 각각 하나씩 가져야하나 ?
        self.embed_target = nn.Embedding(item_num, embed_size) # (m:14586 * d:64), 예측 데이터(p)
        # with open(".\data\Yelp\poi_region_sorted.txt", 'r') as file:
        #     # 모든 행을 읽어와서 첫 번째 열만 리스트로 변환
        #     loaded_embed = [int(line.split('\t')[1].strip()) for line in file.readlines()]
        self.embed_region = nn.Embedding(region_embed_size, embed_size) # 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss() # binary cross entropy

        # Attention을 위한 MLP Layer 생성
        self.attn_layer1 = nn.Linear(embed_size * 2, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

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
        #print("########################### def attention network start ##################################")
        #print(len(user_history), user_history)
        #print(len(target_item), target_item)

        history_ = self.embed_history(user_history) # (b * d)
        region = self.embed_region(history_region) # 
        history = torch.cat((history_, region), -1)
        
        target_ = self.embed_target(target_item) # (b * d)
        target_region_ = self.embed_region(target_region)
        target =  torch.cat((target_, target_region_),-1)
        
        batch_dim = len(target) # (b)
        
        target = torch.reshape(target,(batch_dim, 1,-1))
        input = history * target # (b*d) * (b*1*d) => (b * b * d)

        attention_result = self.relu(self.attn_layer1(input)) # (b * b * d)
        attention_result = self.attn_layer2(attention_result) # (b * b * 1) => 여기서 갑자기 왜 다 같은값 되노?

        exp_A = torch.exp(attention_result) # (b * b * 1) 
        exp_A = exp_A.squeeze(dim=-1)# (b * b)

        mask = self.get_mask(user_history,target_item) # (b * b)
        exp_A = exp_A * mask # (b * b)
        exp_sum = torch.sum(exp_A,dim=-1) # (b)
        exp_sum = torch.pow(exp_sum, self.beta) # (b)

        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * b)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * b * 1)
        result = history * attn_weights# (b * b * d) => 여기서 갑자기 왜 0번 인덱스는 0000??
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)

        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * b * 1) -> (b * b)
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

class NAIS_region_distance_Embedding(nn.Module):
    def __init__(self, item_num, embed_size, hidden_size, beta, region_embed_size, dist_embed_size): # embed_size : 64, beta : 0.5, region_embed_size : 152 dist_embed_size : 1
        super(NAIS_region_distance_Embedding, self).__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.attn_layer1 = nn.Linear(embed_size * 2, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

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
        b: batch size (= input_item_num)
        h: history size (h * 5 = item_num = batch_size)
        d: embedding size
        """
        #print("########################### def attention network start ##################################")
        #print(len(user_history), user_history)
        #print(len(target_item), target_item)

        history_ = self.embed_history(user_history) #
        region = self.embed_region(history_region) # 
        history = torch.cat((history_, region), -1)
        
        target_ = self.embed_target(target_item) #
        target_region_ = self.embed_region(target_region)
        target =  torch.cat((target_, target_region_),-1)
        
        batch_dim = len(target) #
        target = torch.reshape(target,(batch_dim, 1,-1))
        
        input = history * target # 
        attention_result = self.relu(self.attn_layer1(input)) # 
        attention_result = self.attn_layer2(attention_result) # 
        
        dist_embedding_weights = self.embed_distance(torch.tensor(0).to(self.DEVICE)) # d
        target_distance = target_distance.unsqueeze(1) # b => b * 1
        distance = torch.sum(target_distance * dist_embedding_weights, dim = 1)
        distance = distance.unsqueeze(1).unsqueeze(2) # b => b * 1 * 1
        
        exp_input = attention_result + distance # b * d * 1 + b * 1 * 1 = b * d * 1
        
        exp_A = torch.exp(exp_input) # (b * b * 1) 
        exp_A = exp_A.squeeze(dim=-1)# (b * b)

        mask = self.get_mask(user_history, target_item) # (b * b)
        exp_A = exp_A * mask # (b * b)
        exp_sum = torch.sum(exp_A,dim=-1) # (b)
        exp_sum = torch.pow(exp_sum, self.beta) # (b)

        attn_weights = torch.divide(exp_A.T,exp_sum).T # (b * b)
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])# (b * b * 1)
        result = history * attn_weights# (b * b * d) => 여기서 갑자기 왜 0번 인덱스는 0000??
        target = target.reshape([batch_dim,-1,1]) # (b * d * 1)

        prediction = torch.bmm(result, target).squeeze(dim=-1) # (b * b * 1) -> (b * b)
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
