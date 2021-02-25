import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        uu_train_file = path + '/train_u_t10.txt'
        ii_train_file = path + '/train_i_t10.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        #read train_file
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        #n_test: interactions of all test nodes
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.uu_R = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        self.ii_R = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}  # train and test data for items(per user)

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        with open(uu_train_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                for i in train_items:
                    self.uu_R[uid, i] = 1.

        with open(ii_train_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                for i in train_items:
                    self.ii_R[uid, i] = 1.

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            uu_adj_mat = sp.load_npz(self.path + '/uus_adj_mat.npz')
            uu_norm_adj_mat = sp.load_npz(self.path + '/uus_norm_adj_mat.npz')
            uu_mean_adj_mat = sp.load_npz(self.path + '/uus_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat, uu_adj_mat, uu_norm_adj_mat, uu_mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

            sp.save_npz(self.path + '/uus_adj_mat.npz', uu_adj_mat)
            sp.save_npz(self.path + '/uus_norm_adj_mat.npz', uu_norm_adj_mat)
            sp.save_npz(self.path + '/uus_mean_adj_mat.npz', uu_mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat, uu_adj_mat, uu_norm_adj_mat, uu_mean_adj_mat

    # create three adjacency matrix
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        uu_adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        uu_adj_mat = uu_adj_mat.tolil()

        R = self.R.tolil()
        uu_R = self.uu_R.tolil()
        ii_R = self.ii_R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        uu_adj_mat[:self.n_users, :self.n_users] = uu_R
        uu_adj_mat[self.n_users:self.n_items+self.n_users, self.n_users:self.n_items+self.n_users] = ii_R

        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        uu_norm_adj_mat = normalized_adj_single(uu_adj_mat + sp.eye(uu_adj_mat.shape[0]))
        uu_mean_adj_mat = normalized_adj_single(uu_adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(),uu_adj_mat.tocsr(), uu_norm_adj_mat.tocsr(), uu_mean_adj_mat.tocsr()

    # sample training data
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # sample num positive items for user u
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def _get_all_kg_data(self, norm_adj):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        lap = norm_adj.tocoo()
        all_h_list = list(lap.row)
        all_t_list = list(lap.col)
        all_v_list = list(lap.data)
        all_r_list = [0] * len(all_h_list)

        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[], [], []]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])

        all_h_list, all_t_list, all_r_list, all_v_list = new_h_list, new_t_list, new_r_list, new_v_list
        for i in range(len(all_h_list)):
            all_r_list[i] = 0
        return all_h_list, all_t_list, all_r_list, all_v_list

    def _get_all_kg_data_uu(self, norm_adj):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        lap = norm_adj.tocoo()
        all_h_list = list(lap.row)
        all_t_list = list(lap.col)
        all_v_list = list(lap.data)
        all_r_list = [0] * len(all_h_list)

        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[], [], []]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])

        all_h_list, all_t_list, all_r_list, all_v_list = new_h_list, new_t_list, new_r_list, new_v_list
        for i in range(len(all_h_list)):
            all_r_list[i] = 1
        return all_h_list, all_t_list, all_r_list, all_v_list

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
