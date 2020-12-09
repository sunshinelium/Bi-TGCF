import random
import math
import numpy as np
import os
def dict_add(data_dic,keys,values):
    if keys in data_dic.keys():
        data_dic[keys].append(values)
    else:
        data_dic[keys] = [values]
def list_str(list):
    return [str(i) for i in list ]

def read_from_file(file):
    data = []
    n_items = 0
    with open(file, 'r') as f:
        for line in f:
            if line[:-1]:
                lines = line[:-1].split("\t")
                user = int(lines[0])
                movie = int(lines[1])
                score = float(lines[2])
    #            sumRate=sumRate+score
    #            time = int(lines[3])
                data.append([user, movie,score])
                if movie>n_items:
                    n_items = movie
    f.close()
    return data,n_items
def build_data_dic(data):
    data_dic = {}
    for element in data:
        user_id,item_id,rating = element
        dict_add(data_dic,user_id,item_id)
    return data_dic

def split_data(data_dic,filepath,split_ratio=0.8):
    train_file = filepath+'train.txt'
    test_file = filepath+'test.txt'
    train_dic = {}
    test_dic = {}
    train_f = open(train_file,'w+')
    test_f = open(test_file,'w+')
    for user_id in data_dic.keys():
        items = data_dic[user_id]
        random.shuffle(items)
        split_index = math.floor(len(items)*split_ratio)
        dict_add(train_dic,user_id,items[:split_index])
        train_f.write('\t'.join(list_str([user_id]+items[:split_index]))+'\n')
        dict_add(test_dic,user_id,items[split_index:])
        test_f.write('\t'.join(list_str([user_id]+items[split_index:]))+'\n')
    train_f.close()
    test_f.close()
    return train_dic,test_dic

def split_loo(data_dic,filepath):
    train_file = filepath+'/train.txt'
    test_file = filepath+'/test.txt'
    train_dic = {}
    test_dic = {}
    train_f = open(train_file,'w+')
    test_f = open(test_file,'w+')
    for user_id in data_dic.keys():
        items = data_dic[user_id]
        random.shuffle(items)
        split_index = -1
        items_train = items[:split_index]
        if len(items_train)>0:
            train_dic[user_id]=items_train
            for i in items_train:
                train_f.write('\t'.join(list_str([user_id]+[i]+['1']))+'\n')
        test_dic[user_id] = items[split_index:]
        test_f.write('\t'.join(list_str([user_id]+items[split_index:]+['1']))+'\n')
    train_f.close()
    test_f.close()
    return train_dic,test_dic
def split_sparsy(train_dic,filepath,k_keep):
    train_f = open(filepath+'train_'+str(k_keep)+'.txt','w+')
    for user_id in train_dic.keys():
        items = train_dic[user_id]
        split_index = math.ceil(len(items)*k_keep)
        for i in items[:split_index]:
            train_f.write('\t'.join(list_str([user_id]+[i]+['1']))+'\n')
    train_f.close()

def get_test_neg(train_dic,test_dic,neg_num,filepath):
    test_neg = filepath + '/test_neg.txt'
    test_neg_f = open(test_neg,'w+')
    for u in test_dic.keys():
        pos_items = train_dic.get(u,[])
        neg_items = []
        while True:
            if len(neg_items) == neg_num: break
            neg_id = random.randint(0, n_items)
            if neg_id not in train_dic.get(u,[]) and neg_id not in neg_items:
                neg_items.append(neg_id)
        line = str((u,test_dic[u][0]))
        line = line +'\t'+'\t'.join(list_str(neg_items))
        test_neg_f.write(line+'\n')
    test_neg_f.close()

filepath = './cell_sport/'
# dataset = 'cloth'
# save_folder = filepath+ 'beauty_cloth/'
split_init = 0

data,n_items = read_from_file(filepath+'new_reindex.txt')
data = sorted(data,key=lambda x:x[0])
data_dic = build_data_dic(data)
train_dic,test_dic = split_loo(data_dic,filepath)
get_test_neg(train_dic,test_dic,99,filepath)
print('finished')




