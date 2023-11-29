from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import numpy as np
import random
import csv
from haversine import haversine
def get_region(place_coords,size,path):
    """
    place_coords: list, index:poi_id value:(poi_lat, poi_lng)
    size: int, region size(m)
    """
    latitude_max = -55000
    latitude_min =55000
    longitude_max = -55000
    longitude_min = 55000
    places = []
    for lid in range(len(place_coords)):
        now_lat = place_coords[lid][0]
        now_lng = place_coords[lid][1]
        latitude_min = min(latitude_min, now_lat)
        latitude_max = max(latitude_max, now_lat)
        
        longitude_min = min(longitude_min, now_lng)
        longitude_max = max(longitude_max, now_lng)
        places.append((lid,now_lat,now_lng))
    
    print("lat_max:{latmax} lat_min:{latmin}".format(latmax=latitude_max, latmin=latitude_min))
    print("lng_max:{lngmax} lng_min:{lngmin}".format(lngmax=longitude_max, lngmin=longitude_min))

    width1 = haversine((latitude_max,longitude_max),(latitude_max,longitude_min), unit="m")
    width2 = haversine((latitude_min,longitude_max),(latitude_min,longitude_min), unit="m")
    
    height1 = haversine((latitude_max,longitude_max),(latitude_min,longitude_max), unit="m")
    height2 = haversine((latitude_max,longitude_min),(latitude_min,longitude_min), unit="m")

    #지역 쪼개기
    colnum = int((width2+width1)/2/size)
    rownum = int(height1/size)
    print("row: {} col: {}".format(rownum, colnum))
    delta = (longitude_max - longitude_min)/colnum
    areaRangeArr = np.zeros((rownum, colnum, 4))
    areaBusinessArr = dict()
    for i in range(rownum):
        areaBusinessArr[i]=dict()
        for j in range(colnum):
            areaBusinessArr[i][j]=dict()

    place_region = np.zeros([len(place_coords)])-1
    for i in range(rownum):
        print("row{}".format(i))
        lat_min = latitude_min + (latitude_max - latitude_min)/rownum*(i)
        lat_max = latitude_min + (latitude_max - latitude_min)/rownum*(i+1)
        target = [x for x in places if x[1]>=lat_min and x[1]<=lat_max]
        for j in range(colnum):
            lng_min = longitude_min + delta * j
            lng_max = longitude_min + delta * (j+1)

            areaRangeArr[i][j][0] = lat_min
            areaRangeArr[i][j][1] = lat_max
            areaRangeArr[i][j][2] = lng_min
            areaRangeArr[i][j][3] = lng_max

            for lid in target:
                if place_region[lid[0]]>=0:
                    continue
                now_lat = place_coords[lid[0]][0]
                now_lng = place_coords[lid[0]][1]
                if j == colnum-1 and i == rownum-1:
                    if now_lng<=lng_max and now_lat <= lat_max:
                        place_region[lid[0]] = colnum*i + j
                    
                elif j == colnum-1:
                    if now_lng<=lng_max and now_lat < lat_max:
                        place_region[lid[0]] = colnum*i + j

                elif i == rownum-1:
                    if now_lng < lng_max and now_lat <= lat_max:
                        place_region[lid[0]] = colnum*i + j

    f= open(path+"poi_region.txt","w")
    for i in range(len(place_region)):
        f.write("{}\t{}\n".format(i,int(place_region[i])))
    f.close()

def train_test_split_with_time(place_list, freq_list, time_list, test_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i],time_list[i], freq_list[i]))
    
    li.sort(key=lambda x:-x[1])
    test = li[:int(len(li)*test_size)]
    train = li[int(len(li)*test_size):]
    random.shuffle(train)

    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[2])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[2])

    return train_place, test_place, train_freq, test_freq
def get_region_num(path):
    input_file = 'poi_region.txt'
    output_file = 'poi_region_sorted.txt'


    # CSV 파일을 읽고 데이터를 리스트로 저장
    data = []
    with open(path+input_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
    #data.append(header)
        for row in csv_reader:
            row = [int(row[0]), int(row[1])]
            data.append(row)

    data.sort(key=lambda x: x[1])  # region 기준으로 데이터를 정렬

    idx = 0 # region을 0번부터 매핑시키기 위한 새로운 리스트 생성
    before_region_id = data[0][1] # 맨 첫번째 buid 기준
    new_data = []
    for i in data: # 모든 데이터에 대하여
        if i[1] == before_region_id: #buid가 같다면
            new_data.append([i[0], idx]) 
        else:
            idx += 1
            before_region_id = i[1]
            new_data.append([i[0], idx])

    new_data.sort(key=lambda x: x[0])  # user_id 순으로 정렬
    ma = max(new_data, key=lambda x:x[1])
    # 정렬된 데이터를 새로운 파일에 저장
    with open(path+output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerows(new_data)

    print("region num: {}".format(ma[1]+1))
    return ma[1]+1
class Yelp(object):
    def __init__(self):
        # self.user_num = 15359
        self.user_num = 500
        self.poi_num = 14586
        self.directory_path = './data/Yelp/'
        self.poi_file = 'Yelp_poi_coos.txt'
        # self.checkin_file = 'Yelp_checkins.txt'
        self.checkin_file = 'sample.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] ==0 or sparse_raw_time_matrix[uid,lid] >= time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        test_positive = []
        test_negative = []
        pois = np.arange(self.poi_num)
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data
            train_place, test_place, train_freq, test_freq = train_test_split(place_list, freq_list, test_size=test_size, random_state=random_seed)
            # train_place, test_place, train_freq, test_freq = train_test_split_with_time(place_list, freq_list, time_list, test_size)

            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]
            test_positive.append(test_place.tolist())

            negative = list(set(pois) - set(raw_matrix.getrow(user_id).indices))
            random.shuffle(negative)
            # train_negative.append(negative[int(len(negative)*test_size):])
            test_negative.append(negative[:int(len(negative)*test_size)])
        # sparse.save_npz('./data/Foursquare/train_matrix.npz', train_matrix)

        return train_matrix.tocsr(), test_positive, test_negative

    def read_poi_coos(self):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])

        return place_coords

    def generate_data(self, random_seed=0):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, test_negative = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos()
        return train_matrix,  test_positive, test_negative, place_coords

class Foursquare(object):
    def __init__(self):
        self.user_num = 24941
        self.poi_num = 28593
        self.directory_path = './data/Foursquare/'
        self.checkin_file = 'Foursquare_checkins.txt'
        self.poi_file = 'Foursquare_poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] ==0 or sparse_raw_time_matrix[uid,lid] >= time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        test_positive = []
        test_negative = []
        pois = np.arange(self.poi_num)
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data
            # train_place, test_place, train_freq, test_freq = train_test_split(place_list, freq_list, test_size=test_size, random_state=random_seed)
            train_place, test_place, train_freq, test_freq = train_test_split_with_time(place_list, freq_list, time_list, test_size)

            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]
            test_positive.append(test_place)

            negative = list(set(pois) - set(raw_matrix.getrow(user_id).indices))
            random.shuffle(negative)
            # train_negative.append(negative[int(len(negative)*test_size):])
            test_negative.append(negative[:int(len(negative)*test_size)])
        # sparse.save_npz('./data/Foursquare/train_matrix.npz', train_matrix)

        return train_matrix.tocsr(), test_positive, test_negative

    def read_poi_coos(self):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])

        return place_coords

    def generate_data(self, random_seed=0):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, test_negative = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos()
        return train_matrix,  test_positive, test_negative, place_coords

if __name__ == '__main__':
    train_matrix, test_positive, test_negative, place_coords= Yelp().generate_data()
    print(train_matrix.shape, len(test_positive), len(place_coords))


