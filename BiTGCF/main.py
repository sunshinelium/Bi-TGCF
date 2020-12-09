from utility.helper import *
from utility.batch_test import test
from utility.parser import parse_args
from utility.load_data import *
from BiTGCF import BiTGCF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import sys
from random import randint,random
from tqdm import tqdm

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

def get_adj_mat(config,data_generator,adj_type,domain_type):
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if adj_type == 'plain':
        config['norm_adj_%s'%domain_type] = plain_adj
        print('%s use the plain adjacency matrix'%domain_type)

    elif adj_type == 'norm':
        config['norm_adj_%s'%domain_type] = norm_adj
        print('%s use the normalized adjacency matrix'%domain_type)

    elif adj_type == 'gcmc':
        config['norm_adj_%s'%domain_type] = mean_adj
        print('%s use the gcmc adjacency matrix'%domain_type)

    else:
        config['norm_adj_%s'%domain_type] = mean_adj + sp.eye(mean_adj.shape[0])
        print('%s use the mean adjacency matrix'%domain_type)

def get_pretrain_ret(data_generator,test_user_list,data_status,data_type,Ks,BATCH_SIZE,layer_size):
    hr,ndcg = test(sess, model, data_generator,test_user_list,data_type,BATCH_SIZE,Ks, layer_size)
    best_hr = hr[-1]
    best_ndcg = ndcg[-1]
    pretrain_ret = 'pretrained model hit=%s,' \
                    'ndcg=%s' % \
                    (str(['%.4f'%i for i in hr]),
                    str(['%.4f'%i for i in ndcg]))
    pprint(pretrain_ret,save_log_file)
    return hr,ndcg,best_hr,best_ndcg

def print_test_result(hr,ndcg,train_time,test_time,domain_type,data_status):
    if args.verbose > 0:
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.4f=%.4f + %.4f], hit=%s, ndcg=%s at %s' % \
                (epoch, train_time, test_time, losses, loss_source,loss_target , str(['%.4f'%i for i in hr]),
                    str(['%.4f'%i for i in ndcg]),data_status)
        pprint(perf_str,save_log_file)

def find_best_epoch(ndcg_loger,hit_loger,domain_type):
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    hit_10 = hit[:,-1]
    best_rec_0 = max(hit_10)
    idx = list(hit_10).index(best_rec_0)
    pprint('{:*^40}'.format(domain_type+' part'),save_log_file)
    final_perf = "Best Iter=[%d]@[%.1f]\t hit=%s, ndcg=%s" % \
                 (idx-1, time() - t0, str(['%.4f'%i for i in list(hit[idx])]), str(['%.4f'%i for i in list(ndcgs[idx])]))
    pprint(final_perf,save_log_file)
    return final_perf

if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    weight_id = random()
    save_log_dir = './logs/%s/%s/'%(args.dataset,str(args.layer_size))
    ensureDir(save_log_dir)
    save_log_file = open(save_log_dir+'lr_%s_b%s_id_%.4f.txt'%(str(args.lr),args.batch_size,weight_id),'w+')
    config = dict()
    Ks = eval(args.Ks)
    layer_size = args.layer_size
    BATCH_SIZE = args.batch_size
    neg_num = args.neg_num
    source_name,target_name = args.dataset.split('_')
    data_generator_s = Data(path=args.data_path + args.dataset, batch_size=args.batch_size,neg_num=neg_num)
    data_generator_t = Data(path=args.data_path + target_name+'_'+source_name, batch_size=args.batch_size,neg_num=neg_num)
    pprint('{:*^40}'.format('source data info' ),save_log_file)
    data_generator_s.print_statistics(save_log_file)
    pprint('{:*^40}'.format('target data info' ),save_log_file)
    data_generator_t.print_statistics(save_log_file)
    assert data_generator_s.n_users == data_generator_t.n_users ,'data-erro,user should be shared'

    '''compute domain_adj '''
    domain_adj = sp.dok_matrix((data_generator_s.n_users,2),dtype=np.float32)
    domain_adj = domain_adj.tolil()
    R_s = data_generator_s.get_R_mat()
    R_t = data_generator_t.get_R_mat()
    domain_adj[:,0] = R_s.sum(1)
    domain_adj[:,1] = R_t.sum(1)
    domain_adj = domain_adj.todok()
    degree_sum = np.array(domain_adj.sum(1))
    d_inv = np.power(degree_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.
    # d_inv = np.squeeze(d_inv,axis=0)
    d_mat_inv = sp.diags(d_inv[:,0])
    norm_domain_adj = d_mat_inv.dot(domain_adj)
    config['domain_adj'] = np.array(norm_domain_adj.todense())


    config['n_users'] = data_generator_s.n_users
    config['n_items_s'] = data_generator_s.n_items
    config['n_items_t'] = data_generator_t.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    get_adj_mat(config,data_generator_s,args.adj_type,'s')
    get_adj_mat(config,data_generator_t,args.adj_type,'t')

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

        #get sparcy station
    if args.sparcy_flag:
        split_ids_s,split_status_s = data_generator_s.get_sparsity_split()
        split_ids_t,split_status_t = data_generator_t.get_sparsity_split()

    else:
        split_ids_s,split_ids_t,split_status_s,split_status_t=[],[],[],[]
    split_ids_s.append(range(data_generator_s.n_users))
    split_ids_t.append(range(data_generator_t.n_users))
    split_status_s.append('full rating, #user=%d'%data_generator_s.n_users)
    split_status_t.append('full rating, #user=%d'%data_generator_t.n_users)

    model = BiTGCF(data_config=config, args=args,pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        n_layer = len(eval(args.layer_size))
        
        weights_save_path = '%sweights/%s_%s/%s/N_layer=%s/l%s_b%s_layer%s_adj%s_connect%s_drop%s.ckpt' % (args.weights_path, args.dataset,args.keep_ratio, args.layer_fun, layer,
                                                            str(args.lr), args.batch_size,layer,args.adj_type,args.connect_type, args.mess_dropout)
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess = tf.Session()

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s_%s/%s/N_layer=%s/l%s_b%s_layer%s_adj%s_connect%s_drop%s.ckpt' % (args.weights_path, args.dataset,args.keep_ratio, args.layer_fun, layer,
                                                            str(args.lr), args.batch_size,layer,args.adj_type,args.connect_type, args.mess_dropout)

        pprint(pretrain_path,save_log_file)
        try:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, pretrain_path)
            pprint('load the pretrained model parameters from: %s'% pretrain_path,save_log_file)

            # *********************************************************
            # get the performance from pretrained model.
        except:
            sess.run(tf.global_variables_initializer())
            pprint('load fiared , without pretraining.',save_log_file)

    else:
        sess.run(tf.global_variables_initializer())
        pprint('without pretraining.',save_log_file)
        
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    pre_loger_t, rec_loger_t, ndcg_loger_t, hit_loger_t = [], [], [], [] 

    pprint('\n{:*^40}'.format('source initial result'),save_log_file)
    for test_user_list_s,data_status in zip(split_ids_s,split_status_s):
        hr_s,ndcg_s,best_hr_s,best_ndcg_s = get_pretrain_ret(data_generator_s,test_user_list_s,data_status,'source',Ks,BATCH_SIZE,layer_size)
    pprint('\n{:*^40}'.format('target initial result'),save_log_file)
    for test_user_list_t,data_status in zip(split_ids_t,split_status_t):
        hr_t,ndcg_t,best_hr_t,best_ndcg_t= get_pretrain_ret(data_generator_t,test_user_list_t,data_status,'target',Ks,BATCH_SIZE,layer_size)
    ndcg_loger.append(ndcg_s)
    hit_loger.append(hr_s)

    ndcg_loger_t.append(ndcg_t)
    hit_loger_t.append(hr_t)
    if args.save_flag:
        save_saver.save(sess, weights_save_path)
        pprint('save the weights in path: %s'% weights_save_path,save_log_file)
    """
    *********************************************************
    Train.
    """
    stopping_step = 0
    stopping_step_s = 0
    should_stop_s = False
    should_stop_t = False
    verbose = 10
    isonebatch = False
    for epoch in range(args.epoch):
        t1 = time()
        loss, loss_source, loss_target = [],[],[]
        user_input_s,item_input_s,label_s = data_generator_s.get_train_instance()
        user_input_t,item_input_t,label_t = data_generator_t.get_train_instance()
        train_len_s = len(user_input_s)
        train_len_t = len(user_input_t)
        shuffled_idx_s = np.random.permutation(np.arange(train_len_s))
        train_u_s = user_input_s[shuffled_idx_s]
        train_i_s = item_input_s[shuffled_idx_s]
        train_r_s = label_s[shuffled_idx_s]
        shuffled_idx_t = np.random.permutation(np.arange(train_len_t))
        train_u_t = user_input_t[shuffled_idx_t]
        train_i_t = item_input_t[shuffled_idx_t]
        train_r_t = label_t[shuffled_idx_t]
        n_batch_s = train_len_s // args.batch_size + 1
        n_batch_t = train_len_t // args.batch_size + 1
        n_batch_max = max(n_batch_s,n_batch_t)
        n_batch_min = min(n_batch_s,n_batch_t)
        
        if n_batch_s>=n_batch_t:
            pprint('source domain single train',save_log_file)
            # bar_s = ProgressBar('train_source', max=n_batch_max-n_batch_min)
            for i in tqdm(range(n_batch_min,n_batch_max),desc='train_source',ascii=True):
                # bar_s.next()
                min_idx = i*BATCH_SIZE
                max_idx = np.min([(i+1)*BATCH_SIZE,train_len_s])
                if max_idx<(i+1)*BATCH_SIZE:
                    idex = list(range(min_idx,max_idx))+list(np.random.randint(0,train_len_s,(i+1)*BATCH_SIZE-max_idx))
                    train_u_batch = train_u_s[idex]
                    train_i_batch = train_i_s[idex]
                    train_r_batch = train_r_s[idex]
                else:
                    train_u_batch = train_u_s[min_idx: max_idx]
                    train_i_batch = train_i_s[min_idx: max_idx]
                    train_r_batch = train_r_s[min_idx: max_idx]
                _, batch_loss_source= sess.run([model.opt_s, model.loss_source],
                                feed_dict={model.users_s: train_u_batch, model.items_s: train_i_batch,model.label_s:train_r_batch,
                                            model.node_dropout: eval(args.node_dropout),
                                            model.mess_dropout: eval(args.mess_dropout),
                                            model.isTraining:False})
                loss_source.append(batch_loss_source) 
        else:
            pprint('target domain single train',save_log_file)
            # bar_t = ProgressBar('train_target', max=n_batch_max-n_batch_min)
            for i in tqdm(range(n_batch_min,n_batch_max),desc='train_target',ascii=True):
                # bar_t.next()
                min_idx = i*BATCH_SIZE
                max_idx = np.min([(i+1)*BATCH_SIZE,train_len_t])
                if max_idx<(i+1)*BATCH_SIZE:
                    idex = list(range(min_idx,max_idx))+list(np.random.randint(0,train_len_t,(i+1)*BATCH_SIZE-max_idx))
                    train_u_batch = train_u_t[idex]
                    train_i_batch = train_i_t[idex]
                    train_r_batch = train_r_t[idex]
                else:
                    train_u_batch = train_u_t[min_idx: max_idx]
                    train_i_batch = train_i_t[min_idx: max_idx]
                    train_r_batch = train_r_t[min_idx: max_idx]
                _, batch_loss_target= sess.run([model.opt_t, model.loss_target],
                                feed_dict={model.users_t: train_u_batch, model.items_t: train_i_batch,model.label_t:train_r_batch,
                                            model.node_dropout: eval(args.node_dropout),
                                            model.mess_dropout: eval(args.mess_dropout),
                                            model.isTraining:False})
                loss_target.append(batch_loss_target)
        for i in tqdm(range(n_batch_min),desc='train_join',ascii=True):
            # bar_join.next()
            min_idx = i*BATCH_SIZE
            max_idx = np.min([(i+1)*BATCH_SIZE,min([train_len_s,train_len_t])])
            if max_idx<(i+1)*BATCH_SIZE:
                idex = list(range(min_idx,max_idx))+list(np.random.randint(0,min([train_len_s,train_len_t]),(i+1)*BATCH_SIZE-max_idx))
                train_u_batch_s = train_u_s[idex]
                train_i_batch_s = train_i_s[idex]
                train_r_batch_s = train_r_s[idex]                
                train_u_batch_t = train_u_t[idex]
                train_i_batch_t = train_i_t[idex]
                train_r_batch_t = train_r_t[idex]
            else:
                train_u_batch_s = train_u_s[min_idx: max_idx]
                train_i_batch_s = train_i_s[min_idx: max_idx]
                train_r_batch_s = train_r_s[min_idx: max_idx]
                train_u_batch_t = train_u_t[min_idx: max_idx]
                train_i_batch_t = train_i_t[min_idx: max_idx]
                train_r_batch_t = train_r_t[min_idx: max_idx]
            _, batch_loss,batch_loss_source,batch_loss_target= sess.run([model.opt, model.loss,model.loss_source,model.loss_target],
                            feed_dict={model.users_s: train_u_batch_s, model.items_s: train_i_batch_s,model.label_s:train_r_batch_s,
                                        model.users_t: train_u_batch_t, model.items_t: train_i_batch_t,model.label_t:train_r_batch_t,
                                        model.node_dropout: eval(args.node_dropout),
                                        model.mess_dropout: eval(args.mess_dropout),
                                        model.isTraining:True})
            
            loss.append(batch_loss)
            loss_source.append(batch_loss_source)
            loss_target.append(batch_loss_target)
        losses = np.mean(loss)
        loss_source = np.mean(loss_source)
        loss_target = np.mean(loss_target)
        # print("Mean loss in this epoch is: {} , mean source loss is: {},mean target loss is: {}".format(losses,loss_source,loss_target))

        t2 = time()
        #------------------batch_test.py---------------
        pprint('\n'+'{:*^40}'.format('source result'),save_log_file)
        for test_user_list_s,data_status in zip(split_ids_s,split_status_s): 
            hr_s,ndcg_s = test(sess, model, data_generator_s,test_user_list_s,'source',2048, Ks,layer_size)
            print_test_result(hr_s,ndcg_s,t2-t1,time()-t2,'source',data_status)
        t3 = time()
        pprint('\n'+'{:*^40}'.format('target result'),save_log_file)
        for test_user_list_t,data_status in zip(split_ids_t,split_status_t): 
            hr_t,ndcg_t = test(sess, model, data_generator_t,test_user_list_t,'target', 2048,Ks, layer_size)
            print_test_result(hr_t,ndcg_t,t2-t1,time()-t3,'target',data_status)
        t4 = time()



        loss_loger.append(loss)
        ndcg_loger.append(ndcg_s)
        hit_loger.append(hr_s)

        ndcg_loger_t.append(ndcg_t)
        hit_loger_t.append(hr_t)

        best_hr_s, stopping_step_s, should_stop_s = early_stopping(hr_s[-1], best_hr_s,
                                                                    stopping_step_s, flag_step=5)
        best_hr_t,stopping_step, should_stop_t = early_stopping(hr_t[-1], best_hr_t,
                                                                    stopping_step, flag_step=5)


        if all([should_stop_s,should_stop_t]) == True:
            break 


        if hr_t[-1] == best_hr_t and args.save_flag == 1:
            save_saver.save(sess, weights_save_path)
            pprint('save the weights in path:%s'%weights_save_path,save_log_file)
        
        save_log_file.flush()
    final_perf_s = find_best_epoch(ndcg_loger,hit_loger,'source')
    final_perf_t = find_best_epoch(ndcg_loger_t,hit_loger_t,'target')
    save_path = '%soutput/%s.result' % (args.proj_path, args.dataset)
    ensureDir(save_path)
    

    f = open(save_path, 'a')

    f.write(
        '\n lr=%.4f, layer_fun=%s,fuse_type_in =%s,neg_num=%s, n_interaction=%s,connect_type=%s\n\t%s\n%s\n%s\n%s'
        % (args.lr, args.layer_fun, args.fuse_type_in,args.neg_num,
           args.n_interaction,args.connect_type, 'source result',final_perf_s,'target result',final_perf_t))
    f.close()
