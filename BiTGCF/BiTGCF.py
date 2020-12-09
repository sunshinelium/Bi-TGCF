'''
source code
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)

change to Bi-TGCF
paper: Meng Liu et al. Cross Domain Recommendation via Bi-directional Transfer
Graph Collaborative Filtering Networks. In CIKM 2020.

@author: Meng Liu (sunshinel@hust.edu.cn)
'''
import tensorflow as tf
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
# from utility.batch_test import *
class BiTGCF(object):
    def __init__(self, data_config,args, pretrain_data):
        # argument settings
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.initial_type = args.initial_type
        self.pretrain_data = pretrain_data
        self.fuse_type_in = args.fuse_type_in

        self.n_users = data_config['n_users']
        self.n_items_s = data_config['n_items_s']
        self.n_items_t = data_config['n_items_t']

        self.n_fold = 100

        self.norm_adj_s = data_config['norm_adj_s']
        self.norm_adj_t = data_config['norm_adj_t']
        self.n_nonzero_elems_s = self.norm_adj_s.count_nonzero()
        self.n_nonzero_elems_t = self.norm_adj_t.count_nonzero()

        self.domain_laplace = data_config['domain_adj']
        self.connect_way = args.connect_type
        self.layer_fun = args.layer_fun

        self.lr = args.lr
        self.n_interaction = args.n_interaction

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.lambda_s = eval(args.lambda_s)
        self.lambda_t = eval(args.lambda_t) #if None,Variable initial value=0, lambda_s the same condition 

        if self.initial_type == 'x':
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif self.initial_type == 'u':
            self.initializer = tf.random_normal_initializer()

        self.weight_source ,self.weight_target = eval(args.weight_loss)[:]
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users_s = tf.placeholder(tf.int32, shape=(None,))
        self.users_t = tf.placeholder(tf.int32, shape=(None,))
        self.items_s = tf.placeholder(tf.int32, shape=(None,))
        self.items_t = tf.placeholder(tf.int32, shape=(None,))
        self.label_s = tf.placeholder(tf.float32,shape=(None,))
        self.label_t = tf.placeholder(tf.float32,shape=(None,))
        self.isTraining = tf.placeholder(tf.bool)

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights_source = self._init_weights('source',self.n_items_s,None)
        self.weights_target = self._init_weights('target',self.n_items_t,None)
        
        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """
        
        self.ua_embeddings_s, self.ia_embeddings_s,self.ua_embeddings_t, self.ia_embeddings_t = self._create_embed(self.weights_source,
                                                                                                self.weights_target,
                                                                                                self.norm_adj_s,
                                                                                                self.norm_adj_t)



        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings_s = tf.nn.embedding_lookup(self.ua_embeddings_s, self.users_s)
        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users_t)
        self.i_g_embeddings_s = tf.nn.embedding_lookup(self.ia_embeddings_s, self.items_s)
        self.i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.items_t)



        self.scores_s = self.get_scores(self.u_g_embeddings_s,self.i_g_embeddings_s)
        self.scores_t = self.get_scores(self.u_g_embeddings_t,self.i_g_embeddings_t)
        self.mf_loss_s, self.emb_loss_s, self.reg_loss_s  = self.create_cross_loss(self.u_g_embeddings_s,
                                                                          self.i_g_embeddings_s,self.label_s,self.scores_s)
        self.mf_loss_t, self.emb_loss_t, self.reg_loss_t  = self.create_cross_loss(self.u_g_embeddings_t,
                                                                          self.i_g_embeddings_t,self.label_t,self.scores_t)
        self.loss_source = self.mf_loss_s + self.emb_loss_s + self.reg_loss_s
        self.loss_target = self.mf_loss_t + self.emb_loss_t + self.reg_loss_t
        self.loss = self.weight_source * self.loss_source + self.weight_target * self.loss_target
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt_s = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_source,var_list=[self.weights_source])
        self.opt_t = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_target,var_list=[self.weights_target])

    def _init_weights(self,name_scope,n_items,user_embedding):
        all_weights = dict()

        initializer = self.initializer

        if self.pretrain_data is None:
            if user_embedding is None:
                all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding_%s'%name_scope)
                all_weights['item_embedding'] = tf.Variable(initializer([n_items, self.emb_dim]), name='item_embedding_%s'%name_scope)
                print('using xavier initialization')
            else:
                all_weights['user_embedding'] = tf.Variable(initial_value=user_embedding,trainable=True, name='user_embedding_%s'%name_scope)
                all_weights['item_embedding'] = tf.Variable(initializer([n_items, self.emb_dim]), name='item_embedding_%s'%name_scope)
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding_%s'%name_scope, dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding_%s'%name_scope, dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d_%s' %(k,name_scope))
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d_%s' %(k,name_scope))

            all_weights['W_bi_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d_%s' %(k,name_scope))
            all_weights['b_bi_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d_%s' %(k,name_scope))

            all_weights['W_trans_%d' %k] = tf.Variable(
                initializer([2*self.weight_size_list[k+1], self.weight_size_list[k+1]]), name='W_trans_%s_%s' %(k,name_scope))

        return all_weights

    def _split_A_hat(self, X,n_items):
        A_fold_hat = []

        fold_len = (self.n_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X,n_items):
        A_fold_hat = []

        fold_len = (self.n_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def s_t_la2add_layer(self,input_s,input_t,lambda_s,lambda_t,domain_laplace):
        u_g_embeddings_s, i_g_embeddings_s = tf.split(input_s, [self.n_users, self.n_items_s], 0)
        u_g_embeddings_t, i_g_embeddings_t = tf.split(input_t, [self.n_users, self.n_items_t], 0)
        laplace_s = tf.constant(domain_laplace[:,0],name='laplace_s')
        laplace_t = tf.constant(domain_laplace[:,1],name='laplace_t')
        u_g_embeddings_s_lap = tf.transpose(tf.add(laplace_s*tf.transpose(u_g_embeddings_s),laplace_t*tf.transpose(u_g_embeddings_t)))
        u_g_embeddings_t_lap = u_g_embeddings_s_lap
        u_g_embeddings_s_lam = tf.add(lambda_s*u_g_embeddings_s,(1-lambda_s)*u_g_embeddings_t)
        u_g_embeddings_t_lam = tf.add((1-lambda_t)*u_g_embeddings_s,lambda_t*u_g_embeddings_t)
        u_g_embeddings_s = (u_g_embeddings_s_lap+u_g_embeddings_s_lam)/2
        u_g_embeddings_t = (u_g_embeddings_t_lap+u_g_embeddings_t_lam)/2
        ego_embeddings_s = tf.concat([u_g_embeddings_s,i_g_embeddings_s],axis=0)
        ego_embeddings_t = tf.concat([u_g_embeddings_t,i_g_embeddings_t],axis=0)
        return ego_embeddings_s,ego_embeddings_t
    

    def _create_embed(self,weights_s,weights_t,norm_adj_s,norm_adj_t):
        def one_graph_layer_gcf(A_fold_hat,ego_embeddings,weights,k):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            
            ego_embeddings = side_embeddings + tf.multiply(ego_embeddings, side_embeddings)

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            return ego_embeddings   


        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat_s = self._split_A_hat_node_dropout(norm_adj_s,self.n_items_s)
            A_fold_hat_t = self._split_A_hat_node_dropout(norm_adj_t,self.n_items_t)
        else:
            A_fold_hat_s = self._split_A_hat(norm_adj_s,self.n_items_s)
            A_fold_hat_t = self._split_A_hat(norm_adj_t,self.n_items_t)

        ego_embeddings_s = tf.concat([weights_s['user_embedding'], weights_s['item_embedding']], axis=0)
        ego_embeddings_t = tf.concat([weights_t['user_embedding'], weights_t['item_embedding']], axis=0)

        if self.connect_way=='concat':
            all_embeddings_s = [ego_embeddings_s]
            all_embeddings_t = [ego_embeddings_t]
        elif self.connect_way=='mean':
            all_embeddings_s = ego_embeddings_s
            all_embeddings_t = ego_embeddings_t

        for k in range(0, self.n_layers):
            if self.layer_fun == 'gcf':
                ego_embeddings_s = one_graph_layer_gcf(A_fold_hat_s,ego_embeddings_s, weights_s,k)
                ego_embeddings_t = one_graph_layer_gcf(A_fold_hat_t,ego_embeddings_t, weights_t,k)
            if k>=self.n_layers-self.n_interaction and self.n_interaction>0:
                if self.fuse_type_in == 'la2add':
                    ego_embeddings_s,ego_embeddings_t = self.s_t_la2add_layer(ego_embeddings_s,ego_embeddings_t,self.lambda_s,self.lambda_t,self.domain_laplace)

            norm_embeddings_s = tf.math.l2_normalize(ego_embeddings_s, axis=1)
            norm_embeddings_t = tf.math.l2_normalize(ego_embeddings_t, axis=1)

            if self.connect_way=='concat':
                all_embeddings_s += [norm_embeddings_s]
                all_embeddings_t += [norm_embeddings_t]
            elif self.connect_way=='mean':
                all_embeddings_s += norm_embeddings_s
                all_embeddings_t += norm_embeddings_t
        if self.connect_way=='concat':
            all_embeddings_s = tf.concat(all_embeddings_s, 1)
            all_embeddings_t = tf.concat(all_embeddings_t, 1)
        elif self.connect_way=='mean':
            all_embeddings_s = all_embeddings_s/(self.n_layers+1)
            all_embeddings_t = all_embeddings_t/(self.n_layers+1)
        
        u_g_embeddings_s, i_g_embeddings_s = tf.split(all_embeddings_s, [self.n_users, self.n_items_s], 0)
        u_g_embeddings_t, i_g_embeddings_t = tf.split(all_embeddings_t, [self.n_users, self.n_items_t], 0)
        return u_g_embeddings_s, i_g_embeddings_s,u_g_embeddings_t, i_g_embeddings_t


    def get_scores(self,users,pos_items):
        scores = tf.reduce_sum(tf.multiply(users, pos_items),axis = 1)
        return scores
    def create_cross_loss(self, users, pos_items,label,scores):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=scores)

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)



