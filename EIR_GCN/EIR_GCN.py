'''
@author:Liu Fan
'''
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

from utility.helper import *
from utility.batch_test import *

class EIR_GCN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'EIR_GCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.beta = args.beta
        self.alpha = args.alpha
        self.norm_adj = data_config['norm_adj']
        self.uu_norm_adj = data_config['uu_norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        self.all_u_list = data_config['all_u_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']
        self.A_in = data_config['norm_adj']
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)])

        self.all_u_list_uu = data_config['all_u_list_uu']
        self.all_t_list_uu = data_config['all_t_list_uu']
        self.all_v_list_uu = data_config['all_v_list_uu']
        self.A_in_uu = data_config['uu_norm_adj']
        self.A_values_uu = tf.placeholder(tf.float32, shape=[len(self.all_v_list_uu)])

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')

        self.h_uu = tf.placeholder(tf.int32, shape=[None], name='h_uu')
        self.pos_t_uu = tf.placeholder(tf.int32, shape=[None], name='pos_t_uu')

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()
        self._build_model_phase_II()
        if self.alg_type in ['EIR_GCN']:
            self.ua_embeddings, self.ia_embeddings = self._create_eir_embed()


        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        print('using xavier initialization')
     
        self.weight_size_list = [self.emb_dim] + self.weight_size

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        return A_fold_hat

    def _create_eir_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        if self.node_dropout_flag:
            # node dropout.
            uu_A_fold_hat = self._split_A_hat_node_dropout(self.uu_norm_adj)
        else:
            uu_A_fold_hat = self._split_A_hat(self.uu_norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = ego_embeddings

        for k in range(0, self.n_layers):

            temp_embed,uu_temp_embed = [], []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            for f in range(self.n_fold):
                uu_temp_embed.append(tf.sparse_tensor_dense_matmul(uu_A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            uu_side_embeddings = tf.concat(uu_temp_embed, 0)
            # transformed sum messages of neighbors.
            # non-linear activation.
            ego_embeddings = side_embeddings
            uu_ego_embeddings = uu_side_embeddings
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])
            uu_ego_embeddings = tf.nn.dropout(uu_ego_embeddings, 1 - self.mess_dropout[k])
            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)
            uu_norm_embeddings = tf.nn.l2_normalize(uu_ego_embeddings, axis=1)
            all_embeddings = ego_embeddings + self.alpha*norm_embeddings + self.beta * uu_norm_embeddings

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        
        ## In the first version, we implement the bpr loss via the following codes:
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        # In the second version, we implement the bpr loss via the following codes to aviod 'NAN' loss during training:
        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        
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

    def _build_model_phase_II(self):
        self.A_score = self._generate_score(self.h0, self.pos_t0)
        self.A_score_uu = self._generate_score(self.h0_uu, self.pos_t0_uu)
        self.A_out,self.A_out_uu = self._create_attentive_A_out()

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_u_list, self.all_t_list]).transpose()
        A = tf.sparse_softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))

        indices_uu = np.mat([self.all_u_list_uu, self.all_t_list_uu]).transpose()
        A_uu = tf.sparse_softmax(tf.SparseTensor(indices_uu, self.A_values_uu, self.A_in_uu.shape))
        return A, A_uu

    def _generate_score(self, u, t):
        embeddings = tf.concat(
            [self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)
        u_e = tf.nn.embedding_lookup(embeddings, u)
        t_e = tf.nn.embedding_lookup(embeddings, t)
        # transform weight
        u_e = tf.reshape(u_e, [-1, self.emb_dim])
        t_e = tf.reshape(t_e, [-1, self.emb_dim])
        score = tf.reduce_sum(tf.multiply(u_e, t_e), 1)
        return score

    def update_attentive_A(self, sess):
        fold_len = len(self.all_u_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_u_list)
            else:
                end = (i_fold + 1) * fold_len
            feed_dict = {
                self.h0: self.all_u_list[start:end],
                self.pos_t0: self.all_t_list[start:end]
            }
            A_score = sess.run(self.A_score, feed_dict=feed_dict)
            kg_score += list(A_score)
        kg_score = np.array(kg_score)
        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices
        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))

    def update_attentive_A_uu(self, sess):
        kg_score_uu = []
        fold_len = len(self.all_u_list_uu) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_u_list_uu)
            else:
                end = (i_fold + 1) * fold_len
            feed_dict_uu = {
                self.h0_uu: self.all_u_list_uu[start:end],
                self.pos_t0_uu: self.all_t_list_uu[start:end]
            }
            A_score_uu = sess.run(self.A_score_uu, feed_dict=feed_dict_uu)
            kg_score_uu += list(A_score_uu)
        kg_score_uu = np.array(kg_score_uu)
        new_A_uu = sess.run(self.A_out_uu, feed_dict={self.A_values_uu: kg_score_uu})
        new_A_values_uu = new_A_uu.values
        new_A_indices_uu = new_A_uu.indices
        rows_uu = new_A_indices_uu[:, 0]
        cols_uu = new_A_indices_uu[:, 1]
        self.A_in_uu = sp.coo_matrix((new_A_values_uu, (rows_uu, cols_uu)),
                                  shape=(self.n_users + self.n_items, self.n_users + self.n_items))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, uu_plain_adj, uu_norm_adj, uu_mean_adj = data_generator.get_adj_mat()
    all_u_list, all_t_list, all_v_list = data_generator._get_all_kg_data(norm_adj)
    all_u_list_uu, all_t_list_uu, all_v_list_uu = data_generator._get_all_kg_data_uu(uu_norm_adj)
    config['all_u_list'] = all_u_list
    config['all_t_list'] = all_t_list
    config['all_v_list'] = all_v_list
    config['all_u_list_uu'] = all_u_list_uu
    config['all_t_list_uu'] = all_t_list_uu
    config['all_v_list_uu'] = all_v_list_uu
    config['n_relations'] = 2

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        config['uu_norm_adj'] = uu_plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        config['uu_norm_adj'] = uu_norm_adj
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()
    model = EIR_GCN(data_config=config, pretrain_data=None)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss
            if (idx + 1)%(int(n_batch/args.n_split)) == 0:
                model.update_attentive_A(sess)
                model.update_attentive_A_uu(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][1],
                        ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                        ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
