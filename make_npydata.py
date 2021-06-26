import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')


'''please set your dataset path'''
try:
    shanghaiAtrain_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_A_final/train_data/images_crop/'
    shanghaiAtest_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_A_final/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully", len(train_list), len(test_list))
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")


try:
    shanghaiBtrain_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_B_final/train_data/images_crop/'
    shanghaiBtest_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_B_final/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully", len(train_list), len(test_list))
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")
