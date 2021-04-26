# @Time : 2020/12/4 10:20
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : Prototype.py
import os

import numpy as np

from FuncTool import load_file


class MIL:
    """
        The origin class of MIL, and all given vector data's type must be numpy.array.
        :param
            para_path:
                The path of MIL data sets.
            para_has_ins_label:
                True if the instances have label else False, and the default setting is True.
        @attribute
            data_name:
                The data set name.
            bags:
                The data of MIL bags with label.
            num_bags:
                The number of MIL bags, e.g., the num_bags = 92 for the data set musk1.
            num_classes:
                The number of classes for MIL bags, e.g., the num_classes = 2 for the data set musk1.
            bags_size:
                The size of bags in the MIL data set.
            bags_label:
                The label of bags in the MIL data set.
            num_instances:
                The summary number of instances in the MIL data set.
            dimensions:
                The dimensions of the given data set.
            ins:
                The instances space.
            ins_idx:
                The index for instances space.
            ins_lab:
                Theindex for instances space, and the instances form the same bag have the same index.
        @example:
            # >>> temp_file_name = '../Data/Benchmark/musk1.mat'
            # >>> mil = MIL(temp_file_name)
            # >>> mil.get_info()
        """

    def __init__(self, para_folder_index, para_data):
        """
        The constructor
        """

        self.data_name = ''
        self.bags = []
        self.num_bags = 0
        self.num_classes = 0
        self.bags_size = []
        self.bags_label = []
        self.num_ins = 0
        self.dimensions = 0
        self.ins = []
        self.ins_idx = []
        self.ins_lab = []
        self.path = ''
        self.has_ins_label = True
        self.__initialize(para_folder_index, para_data)

    def __initialize(self, para_folder_index, para_data):
        """
        The initialize for MIL, e.g., load MIL data sets.
        @param:
        ------------
            para_file_name: the file name of MIL data sets.
        ------------
        """
        self.path = self.get_path(para_folder_index, para_data)
        self.bags = load_file(self.path)
        # 输出包的数量，np.shape[0]输出bags的行数目，shape[1]输出列数量
        self.num_bags = np.shape(self.bags)[0]
        # 输出包的大小，并复制
        self.bags_size = np.zeros(self.num_bags, dtype=int)
        self.bags_label = self.bags_size.copy()

        for i in range(self.num_bags):
            self.bags_size[i] = len(self.bags[i, 0])
            self.bags_label[i] = self.bags[i, 1]
        self.num_ins = np.sum(self.bags_size)
        self.dimensions = len(self.bags[0, 0][0])
        self.dimensions = self.dimensions - 1 if self.has_ins_label else self.dimensions
        self.num_classes = len(set(self.bags_label))

        """
        Get the instances space.
        """
        self.ins = np.zeros((self.num_ins, self.dimensions))
        self.ins_idx = np.zeros(self.num_bags + 1, dtype=int)
        self.ins_lab = np.zeros(self.num_ins, dtype=int)
        for i in range(self.num_bags):
            self.ins_idx[i + 1] = self.bags_size[i] + self.ins_idx[i]
            self.ins[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bags[i, 0][:, : self.dimensions]
            self.ins_lab[self.ins_idx[i]: self.ins_idx[i + 1]] = np.tile([i], (1, self.bags_size[i]))

        """
        Processing path.
        """
        temp_para = self.path.split('/')
        self.data_name = temp_para[-1].split('.')[0]

    def get_info(self):
        """
        Get the all information of current data set.
        """
        print("The ", self.data_name, "'s information is:\n"
                                      "Number bags:", self.num_bags, "\n"
                                                                     "Number classes:", self.num_classes, "\n"
                                                                                                          "Bag size:",
              self.bags_size[:5], "...\n"
                                  "Bag label", self.bags_label[:5], "...\n"
                                                                    "Number instances:", self.num_ins, "\n"
                                                                                                       "Instance dimensions:",
              self.dimensions, "\n"
                               "Instance index:", self.ins_idx[: 5], "...\n"
                                                                     "Instance label corresponding bag'S index:",
              self.ins_lab[:5], "...\n"
                                "Does instance label is given?", self.has_ins_label)

    def get_index(self, para_k=10):
        """
        Get the training set index and test set index.
        @param
            para_k:
                The number of k-th fold.
        :return
            ret_tr_idx:
                The training set index, and its type is dict.
            ret_te_idx:
                The test set index, and its type is dict.
        """
        temp_rand_idx = np.random.permutation(self.num_bags)

        temp_fold = int(self.num_bags / para_k)
        ret_tr_idx = {}
        ret_te_idx = {}
        for i in range(para_k):
            temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
            temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
            ret_tr_idx[i] = temp_tr_idx
            ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
        return ret_tr_idx, ret_te_idx

    def get_path(self, para_folder_index, para_data_index):
        """
        Get the path of preprocessed data.
        :param para_folder_index: the folder`s index of preprocessed data.
        :param para_data_index: the path`s index of preprocessed data.
        :return:
        """
        folder = ["../../data/bechmark/nonnormalized",
                  "../../data/bechmark/normalized",
                  "../../data/newsgroups/nonnormalized",
                  "../../data/newsgroups/normalized"]
        file = os.listdir(folder[para_folder_index])
        file_path = []
        for i in range(0, len(file)):
            file_path.append(folder[para_folder_index] + '/' + file[i])
        # print(file_path[para_data_index])
        return file_path[para_data_index]

    def get_classifier(self,para_model):
        if para_model==1:
            return


if __name__ == '__main__':
    mil = MIL(1, 0)
    mil.get_path(1,0)
