""" From https://github.com/maxitron93/stratified_sampling_for_XML """

import random
import numpy as np
from datetime import datetime
from collections import Counter
from operator import itemgetter
from typing import Any

from src.utils.stratify_function import helper_funcs


def stratified_train_test_split(
    X,
    y,
    target_test_size,
    random_state=None,
    epochs=50,
    swap_probability=0.1,
    threshold_proportion=0.1,
    decay=0.1,
):
    if random_state != None:
        random.seed(random_state)

    # To keep track of how long the initialization takes
    start_time = datetime.now()

    # Keep track how how many instances have been swapped to train or test
    swap_counter = {
        "to_train": 0,
        "to_test": 0,
    }

    # 1. Create instances_dict to keep track of instance information:
    # labels: array of labels, []
    # train_or_test: string, 'train' or 'test'
    # instance_score: float, adjusted sum of label scores
    instances_dict = helper_funcs.create_instances_dict(X, y, target_test_size)

    # 1.5 Get average number of labels per instance
    labels_per_instance = []
    for instance_id, instance_dict in instances_dict.items():
        labels_count = len(instance_dict["labels"])
        labels_per_instance.append(labels_count)
    average_labels_per_instance = sum(labels_per_instance) / len(labels_per_instance)

    # 2. Create labels_dict to keep track of label information:
    # train: int, number of times label appears in train set
    # test: int, number of times label appears in test set
    # label_score: float, label score
    labels_dict = helper_funcs.create_labels_dict(instances_dict)

    # 3. Calculate the label score for each label in labels_dict
    # Positive score if too much of the label is in the test set
    # Negative score if too much of the label is in the train set
    helper_funcs.score_labels(
        labels_dict, target_test_size, average_labels_per_instance
    )

    # 4. Calculate the instance score for each instance in instances_dict
    # A high score means the instance is a good candidate for swapping
    helper_funcs.score_instances(instances_dict, labels_dict)

    # 5. Calculate the total score
    # The higher the score, the more 'imbalanced' the distribution of labels between train and test sets
    total_score = helper_funcs.calculate_total_score(instances_dict)
    print(
        f'Starting score: {round(total_score)}. Calculated in {str(datetime.now() - start_time).split(".")[0]}'
    )

    # Main loop to create stratified train-test split
    for epoch in range(epochs):
        # To keep track of how long each itteration takes
        itteration_start_time = datetime.now()

        # 6. Calculate the threshold score for swapping
        threshold_score = helper_funcs.calculte_threshold_score(
            instances_dict,
            average_labels_per_instance,
            epoch,
            threshold_proportion,
            decay,
        )

        # 7. Swap the instances with instance_score that is greater than the threshold score
        # Probability of swapping an instance is swap_probability
        helper_funcs.swap_instances(
            instances_dict,
            threshold_score,
            swap_counter,
            average_labels_per_instance,
            epoch,
            swap_probability,
            decay,
        )

        # 2. Recreate labels_dict with updated train-test split
        labels_dict = helper_funcs.create_labels_dict(instances_dict)

        # 3. Recalculate the label score for each label in labels_dict
        helper_funcs.score_labels(
            labels_dict, target_test_size, average_labels_per_instance
        )

        # 4. Recalculate the instance score for each instance in instances_dict
        helper_funcs.score_instances(instances_dict, labels_dict)

        # 5. Recalculate the total score
        total_score = helper_funcs.calculate_total_score(instances_dict)
        print(
            f'Epoch {epoch + 1}/{epochs} score: {round(total_score)}. Calculated in {str(datetime.now() - itteration_start_time).split(".")[0]}'
        )

    # Prepare X_train, X_test, y_train, y_test
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for instance_id, instance_dict in instances_dict.items():
        if instance_dict["train_or_test"] == "train":
            X_train.append(X[instance_id])
            y_train.append(y[instance_id])
        elif instance_dict["train_or_test"] == "test":
            X_test.append(X[instance_id])
            y_test.append(y[instance_id])
        else:
            print(f"Something went wrong: {instance_id}")

    # Print some statistics
    actual_test_size = len(X_test) / (len(X_train) + len(X_test))
    print(f'To train: {swap_counter["to_train"]}')
    print(f'To test: {swap_counter["to_test"]}')
    print(f"Target test size: {target_test_size}")
    print(f"Actual test size: {actual_test_size}")

    return X_train, X_test, y_train, y_test


def iterative_stratification(
    data: list[Any], labels: list[list[str]], ratios: list[float]
):
    # Implemented by Sotiris Lamprinidis
    data = data.copy()
    labels = labels.copy()
 
    labels_unique = sorted(set([lbl for lbls in labels for lbl in lbls])) 
    label_to_index = {lbl: i for i, lbl in enumerate(labels_unique)} # [icd icdn ] [f1 fn]
    desired_samples = [len(labels_unique) * r for r in ratios]
    #[[32],[10]]
    desired_labels = [[0 for _ in range(len(labels_unique))] for r in ratios]

    sets = [list() for _ in range(len(ratios))]

    lc = Counter(lbl for lbls in labels for lbl in lbls)
   
    for i, label in enumerate(labels_unique):
        num_this = lc[label]
        for j, ratio in enumerate(ratios):
            desired_labels[j][i] = num_this * ratio
            # นับ counticd [icd1,icd2] [icd1,icd2]
    
   
    while labels: #icd Z.13
        label = lc.most_common()[-1][0] # เอาตัวที่ความถี่น้อยที่สุดมาก่อน
        lbl = label_to_index[label]
   
        dataset_label = [
            (i, (x, y)) for i, (x, y) in enumerate(zip(data, labels)) if label in y # เราจะได้แถวที่ มีเฉพาะความถี่ที่เราสนใจ
        ]
        # print(f'-------{len(dataset_label)}--------')
        # print(np.sum(desired_labels[0]))
        # print(np.sum(desired_labels[1]))
        for index, (x, y) in sorted(dataset_label, key=itemgetter(0), reverse=True):
          
            
            desired = sorted(
                enumerate([desired_labels[j][lbl] for j in range(len(ratios))]), # เรียกเอา icd ที่ถูกเลือกมาก มันจะได้ .85 กับ .15 ขอ id นั้นๆ 
                key=itemgetter(1),
                reverse=True,
            ) #[(0, 40), (1, 10)]  .85 กับ .15 
         
            
            if desired[0][1] != desired[1][1]: #ไม่เท่ากันให้เลือก กลุ่ม1 ปกติกลุ่ม 1 จะเรียงมาแล้วเอาตัวที่ % เยอะกว่า
                chosen = desired[0] # เลือก (0, 40) .85
               
            else:
                
                desired = sorted(
                    [
                        (i, desired_samples[i])
                        for i, _ in [x for x in desired if x[1] == desired[0][1]] #  ถ้ากองขวา == ก้องซ้าย จะมี (0,1) (1,1)
                    ],
                    key=itemgetter(1),
                    reverse=True,
                )#เรียงเอาความถี่สูงขึ้นมาก่อน
            
                if desired[0][1] != desired[1][1]: #ถ้ากองสองกองไม่เท่ากันเลือกองตั้งต้นที่มากกว่า
                    chosen = desired[0]
                    
                else: #เท่าให้สุ่ม
                  
                    chosen = random.choice(desired)
                 
            sets[chosen[0]].append(x) #(0,1) เพิ่ม data
            del labels[index] #ลบที่ dataset นะ
            del data[index] #ลบที่ dataset นะ
            #ลบมันทั้งคู่ออกไป 
          
            for label in y:
                l_this = label_to_index[label]
                desired_labels[chosen[0]][l_this] -= 1 # ลดความถี่ที่ถูกแบ่งกลุ่มลงไป 1 หน่วย ที่ desire_label [3,6,9] [1,2,3] => [3,6-1,9][1,2,3]
                lc.subtract({label: 1}) # ลดความถี่ที่ถูกเลือก frequency ลง
                if lc[label] <= 0:
                    del lc[label] #ลบจริง
            desired_samples[chosen[0]] -= 1
    return sets
