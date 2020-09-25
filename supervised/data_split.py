import pandas as pd
import numpy as np

def return_tumor_or_not(dic, one_id):
    return dic[one_id]

def create_dict(train_labels_path):
    df = pd.read_csv(train_labels_path)
    result_dict = {}
    for index in range(df.shape[0]):
        one_id = df.iloc[index,0]
        tumor_or_not = df.iloc[index,1]
        result_dict[one_id] = int(tumor_or_not)
    return result_dict 

def generate_split(train_labels_path, train_WSIlabels_path, train_percent = 0.8, keys_for_cv = None):
    ids = pd.read_csv(train_WSIlabels_path)
    wsi_dict = {}
    for i in range(ids.shape[0]):
        wsi = ids.iloc[i,1]
        train_id = ids.iloc[i,0]
        wsi_array = wsi.split('_')
        number = int(wsi_array[3])
        if wsi_dict.get(number) is None:
            wsi_dict[number] = [train_id]
        else:
            wsi_dict[number].append(train_id)

    wsi_keys = list(wsi_dict.keys())
    np.random.seed()
    np.random.shuffle(wsi_keys)
    amount_of_keys = len(wsi_keys)
    if keys_for_cv == None:
        keys_for_train = wsi_keys[0:int(amount_of_keys*train_percent)]
        keys_for_cv = wsi_keys[int(amount_of_keys*train_percent):]
    else:
        keys_for_train = list(set(wsi_keys)-set(keys_for_cv))

    train_ids = []
    cv_ids = []

    for key in keys_for_train:
        train_ids += wsi_dict[key]

    for key in keys_for_cv:
        cv_ids += wsi_dict[key]

    dic = create_dict(train_labels_path)

    train_labels = []
    cv_labels = []

    train_tumor = 0
    for one_id in train_ids:
        is_tumor = return_tumor_or_not(dic, one_id)
        train_tumor += is_tumor
        train_labels.append(is_tumor)

    cv_tumor = 0
    for one_id in cv_ids:
        is_tumor = return_tumor_or_not(dic, one_id)
        cv_tumor += is_tumor
        cv_labels.append(is_tumor)

    total = len(train_ids) + len(cv_ids)

    print("Amount of train labels: {}, {}/{}".format(len(train_ids), train_tumor, len(train_ids)-train_tumor))
    print("Amount of cv labels: {}, {}/{}".format(len(cv_ids), cv_tumor, len(cv_ids) - cv_tumor))
    print("Percentage of cv labels: {}".format(len(cv_ids)/total))

    return train_ids, cv_ids, train_labels, cv_labels, keys_for_cv #Also return CV keys for proper evaluation after training.

if __name__ == "__main__":
    keys_for_cv = [1, 12, 16, 21, 24, 25, 27, 32, 36, 50, 53, 58, 62, 67, 74, 76, 82, 89, 98, 105, 108, 110, 115, 119, 121, 138, 140, 141, 151, 156]
    train_ids, cv_ids, train_labels, cv_labels, keys_for_cv = generate_split(train_labels_path = "train_labels.csv",
                                                                             train_WSIlabels_path = "patch_id_wsi_full.csv",
                                                                             keys_for_cv = keys_for_cv)