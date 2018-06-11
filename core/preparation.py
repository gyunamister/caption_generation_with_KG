import json
import pickle
import os
import numpy as np
from collections import Counter


def main():
    folder = '../training_data'

    with open('../training_data/objects.json') as data_file:
        objects = json.load(data_file)

    dummy = []
    for image in objects:
        dummy += [row['names'][0] for row in image['objects']]

    object_counter = dict(Counter(dummy))

    set_a = set([k for k,v in object_counter.items() if v>=100])
    list_a = list(set_a)
    list_a.sort()
    print(len(list_a))
    list_a_index = {}
    for i,row in enumerate(list_a):
        list_a_index[row] = i
    not_list = []
    one_hot_label_data = {}

    for image in objects:
        check = False
        dummy = np.zeros(len(set_a))
        for wow in image['objects']:
            if wow['names'][0] in list(set_a):
                check =True
                dummy[list_a_index[wow['names'][0]]] = 1

        if check:
            one_hot_label_data[image['image_id']] = dummy
        else:
            not_list.append(image['image_id'])
    f = open('../resized_training_data/not_list.pkl', 'wb')
    pickle.dump(not_list, f)
    f = open('../resized_training_data/label_annotation.pkl', 'wb')
    pickle.dump(one_hot_label_data, f)

    with open('../training_data/relationships.json') as data_file:
        relationships = json.load(data_file)


    relation_array = []
    for image in relationships:
        for row in image['relationships']:
            if 'names' in row['object']:
                object_name = row['object']['names'][0]
            elif 'name' in row['object']:
                object_name = row['object']['name']
            else:
                print('error_object')
            if 'names' in row['subject']:
                subject_name = row['subject']['names'][0]
            elif 'name' in row['subject']:
                subject_name = row['subject']['name']
            else:
                print('error_subject')
            relation_array.append((object_name, subject_name))
    relation_counter = dict(Counter(relation_array))
    relations = [k for k, v in relation_counter.items() if v>=10]
    dummy = []
    for x in relations:
        dummy += [x[0], x[1]]
    set_b = set(dummy)
    list_b = list(set_b)


    total_list = list_a
    for x in list_b:
        if not x in list_a:
            total_list.append(x)

    total_index_dict = {}
    for i, name in enumerate(list_a):
        total_index_dict[name] = i

    adj_matrix = np.zeros((len(total_list), len(total_list)))
    for row in relations:
        adj_matrix[total_index_dict[row[0]]][total_index_dict[row[1]]] = 1

    resized_folder = '../resized_training_data/adj_matrix/'
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    np.save(resized_folder + 'adj_matrix.npy',adj_matrix)
    print(len(total_list))




if __name__ == '__main__':
    main()