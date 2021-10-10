from omniglot import Omniglot
import os
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')

class OmniglotNshot :

    def __init__(self, batch_size, n_way , k_shot, k_query, h, w):

        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.h = h
        self.w = w
        self.csv_file = 'data.csv'

        if 'omniglot.npy' not in os.listdir():

            data = Omniglot(self.h, self.w, csv_file = self.csv_file)
            temp = dict()
            i = 0
            for (image, label) in data :
                i = i+1
                if label in temp.keys():
                    temp[label].append(image)
                else:
                    temp[label] = [image]


            data = []
            for label, images in temp.items():
                data.append(images)
            data = np.array(data).astype(np.float)
            print("data saved to a numnpy file")
            np.save('omniglot.npy', data)

            print("shape of data is : " + str(data.shape)) # num_classes * 20 * 1 * h * w
            del temp

        else :

            print("data loaded from existing numpy file")
            data = np.load('omniglot.npy')

        self.x_train, self.x_test = data[:1200], data[1200:]
        print("train data shape : "+ str(self.x_train.shape))
        print("test data shape : " + str(self.x_test.shape))

        del data

        self.indexes = {'train': 0, 'test' : 0}
        self.datasets = {'train': self.x_train, 'test': self.x_test}

        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train']),
                               'test': self.load_data_cache(self.datasets['test'])}



    def normalization(self):

        mean = np.mean(self.x_train)
        std = np.std(self.x_train)
        max = np.max(self.x_train)
        min = np.min(self.x_train)

        self.x_train = (self.x_train - mean) / std
        self.x_test = (self.x_test - mean) / std

        mean = np.mean(self.x_train)
        std = np.std(self.x_train)
        max = np.max(self.x_train)
        min = np.min(self.x_train)


    def load_data_cache(self, data):

        # param data : [num_classes * 20 * 1 * h * w]
        # returns    : [support_set_x, support_set_y, target_x, target_y]

        set_size = self.k_shot * self.n_way # size of meta train set
        query_size = self.k_query * self.n_way # size of meta test set
        data_cache = []

        for sample in range(10):

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []

            for i in range(self.batch_size): # one batch is one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                # select n_way classes randomly to create spt and qry
                sel_classes = np.random.choice(data.shape[0], self.n_way, False)

                for j, cls in enumerate(sel_classes):

                    # select image for the class chosen
                    sel_imgs = np.random.choice(20, self.k_shot + self.k_query, False)

                    x_spt.append(data[cls][sel_imgs[:self.k_shot]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    x_qry.append(data[cls][sel_imgs[self.k_shot:]])
                    y_qry.append([j for _ in range(self.k_query)])
                    
                # one set of spt and qry are ready
                perm = np.random.permutation(self.n_way * self.k_shot) # permuting for support
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot,1, self.h, self.w)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query) # permuting for query
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.h, self.w)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # [one_set, 1, h, w] -> [batch_size, one_set, 1, h, w]

                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # x_spts has size [batch_size, set_size, 1, h, w]
            # y_spts has size [batch_size, set_size)
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batch_size, set_size, 1, self.h, self.w)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batch_size, set_size)
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batch_size, query_size, 1, self.h, self.w)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batch_size, query_size)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode):

        # we will extract each complete set of [spts and qrys] * batch_size
        # when we run out of data, we will upload data cache

        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch














                



























obj  = OmniglotNshot(4,1,5,5,5,84,84)






















