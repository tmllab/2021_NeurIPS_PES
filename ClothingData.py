import numpy as np


def read_key_list(file_name):
    keyList = []
    fo = open(file_name, "r")
    print("Read file:", fo.name)
    for line in fo.readlines():
        line = line.strip()
        keyList.append(line)
 
    fo.close()
    return np.array(keyList)


def read_share_list(file_name):
    key_list = []
    noisy_labels_list = []
    clean_labels_list = []
    fo = open(file_name, "r")
    print("Read file:", fo.name)
    for line in fo.readlines():
        strs = line.strip().split(";;")
        if(len(strs) == 3):
            key_list.append(strs[0])
            noisy_labels_list.append(strs[1])
            clean_labels_list.append(strs[2])
        else:
            print(line)
    fo.close()
    return np.array(key_list), np.array(noisy_labels_list, dtype=np.int), np.array(clean_labels_list, dtype=np.int)


def read_kv_map(file_name):
    kv_map = {}
    fo = open(file_name, "r")
    print("Read file:", fo.name)
    for line in fo.readlines():
        line = line.strip()
        kv_strs = line.split(" ")
        if(len(kv_strs) == 2):
            kv_map[kv_strs[0]] = kv_strs[1]
        else:
            print(line)
    fo.close()
    return kv_map


def get_value_from_kv(keys, kvs):
    labels = []
    for key in keys:
        if(key in kvs):
            labels.append(kvs[key])
        else:
            print(key, "dose not exists in kvs...")

    return np.array(labels, dtype=np.int)


save_file = 'data/Clothing1M_Official/Clothing1m-data.npy'
clean_test_key_file = 'data/Clothing1M_Official/annotations/clean_test_key_list.txt'
clean_val_key_file = 'data/Clothing1M_Official/annotations/clean_val_key_list.txt'
noisy_train_key_file = 'data/Clothing1M_Official/annotations/noisy_train_key_list.txt'
noisy_label_kv = 'data/Clothing1M_Official/annotations/noisy_label_kv.txt'
clean_label_kv = 'data/Clothing1M_Official/annotations/clean_label_kv.txt'
share_file = 'data/Clothing1M_Official/annotations/share.txt'


noisy_train_key_list = read_key_list(noisy_train_key_file)
clean_val_key_list = read_key_list(clean_val_key_file)
clean_test_key_list = read_key_list(clean_test_key_file)

print(noisy_train_key_list.shape, clean_val_key_list.shape, clean_test_key_list.shape)

noisy_label_kv_map = read_kv_map(noisy_label_kv)
clean_label_kv_map = read_kv_map(clean_label_kv)

print(len(noisy_label_kv_map), len(clean_label_kv_map))


noisy_labels = get_value_from_kv(noisy_train_key_list, noisy_label_kv_map)
clean_val_labels = get_value_from_kv(clean_val_key_list, clean_label_kv_map)
clean_test_labels = get_value_from_kv(clean_test_key_list, clean_label_kv_map)

print(noisy_labels.shape, clean_val_labels.shape, clean_test_labels.shape)

share_key_list, share_noisy_labels_list, share_clean_labels_list = read_share_list(share_file)
print(share_key_list.shape, share_noisy_labels_list.shape, share_clean_labels_list.shape)

kvDic = {}
kvDic['train_data'] = noisy_train_key_list
kvDic['train_labels'] = noisy_labels
kvDic['clean_val_data'] = clean_val_key_list
kvDic['clean_val_labels'] = clean_val_labels
kvDic['test_data'] = clean_test_key_list
kvDic['test_labels'] = clean_test_labels
kvDic['share_data'] = share_key_list
kvDic['share_noisy_labels'] = share_noisy_labels_list
kvDic['share_clean_labels'] = share_clean_labels_list
np.save(save_file, kvDic, allow_pickle=True)
