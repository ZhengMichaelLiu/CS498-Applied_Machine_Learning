import numpy as np

# Read in data
Dataset = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')

# do 10 times
accuracy = 0
counter = 0
while counter < 10:
    # For each time
    # Random
    np.random.shuffle(Dataset)

    # Split
    train_data = Dataset[: 613] # 80% of total data
    train_pos = train_data[train_data[:, -1] == 1] # all the pos
    train_neg = train_data[train_data[:, -1] == 0] # all the neg

    train_pos_feature = train_pos[:, :-1]
    train_neg_feature = train_neg[:, :-1]

    test_data = Dataset[613: ]
    test_data_feature = test_data[:, : -1]
    test_data_label =  test_data[:, -1]

    # get non-missing feature
    
    attribute_3_non_missing_pos = []
    attribute_4_non_missing_pos = []
    attribute_6_non_missing_pos = []
    attribute_8_non_missing_pos = []

    attribute_3_non_missing_neg = []
    attribute_4_non_missing_neg = []
    attribute_6_non_missing_neg = []
    attribute_8_non_missing_neg = []

    for each_pos_data in train_pos_feature:
        if each_pos_data[2] != 0:
            attribute_3_non_missing_pos.append(each_pos_data[2])
        if each_pos_data[3] != 0:
            attribute_4_non_missing_pos.append(each_pos_data[3])
        if each_pos_data[5] != 0:
            attribute_6_non_missing_pos.append(each_pos_data[5])
        if each_pos_data[7] != 0:
            attribute_8_non_missing_pos.append(each_pos_data[7])
    
    attribute_3_non_missing_pos_array = np.asarray(attribute_3_non_missing_pos)
    attribute_4_non_missing_pos_array = np.asarray(attribute_4_non_missing_pos)
    attribute_6_non_missing_pos_array = np.asarray(attribute_6_non_missing_pos)
    attribute_8_non_missing_pos_array = np.asarray(attribute_8_non_missing_pos)


    for each_neg_data in train_neg_feature:
        if each_neg_data[2] != 0:
            attribute_3_non_missing_neg.append(each_neg_data[2])
        if each_neg_data[3] != 0:
            attribute_4_non_missing_neg.append(each_neg_data[3])
        if each_neg_data[5] != 0:
            attribute_6_non_missing_neg.append(each_neg_data[5])
        if each_neg_data[7] != 0:
            attribute_8_non_missing_neg.append(each_neg_data[7])
    
    attribute_3_non_missing_neg_array = np.asarray(attribute_3_non_missing_neg)
    attribute_4_non_missing_neg_array = np.asarray(attribute_4_non_missing_neg)
    attribute_6_non_missing_neg_array = np.asarray(attribute_6_non_missing_neg)
    attribute_8_non_missing_neg_array = np.asarray(attribute_8_non_missing_neg)

    # calculate class prob
    p_y1 = len(train_pos) / len(train_data)
    p_y0 = 1 - p_y1

    # Pos - each column mean
    pos_mean = []
    for i in range(8):
        if i == 2:
            pos_mean.append(np.mean(attribute_3_non_missing_pos_array))
        elif i == 3:
            pos_mean.append(np.mean(attribute_4_non_missing_pos_array))
        elif i == 5:
            pos_mean.append(np.mean(attribute_6_non_missing_pos_array))
        elif i == 7:
            pos_mean.append(np.mean(attribute_8_non_missing_pos_array))
        else:
            pos_mean.append(np.mean(train_pos_feature[:, i]))
    # Pos - each column std
    pos_std = []
    for i in range(8):
        if i == 2:
            pos_std.append(np.std(attribute_3_non_missing_pos_array))
        elif i == 3:
            pos_std.append(np.std(attribute_4_non_missing_pos_array))
        elif i == 5:
            pos_std.append(np.std(attribute_6_non_missing_pos_array))
        elif i == 7:
            pos_std.append(np.std(attribute_8_non_missing_pos_array))
        else:
            pos_std.append(np.std(train_pos_feature[:, i]))

    # Neg - each column mean
    neg_mean = []
    for i in range(8):
        if i == 2:
            neg_mean.append(np.mean(attribute_3_non_missing_neg_array))
        elif i == 3:
            neg_mean.append(np.mean(attribute_4_non_missing_neg_array))
        elif i == 5:
            neg_mean.append(np.mean(attribute_6_non_missing_neg_array))
        elif i == 7:
            neg_mean.append(np.mean(attribute_8_non_missing_neg_array))
        else:
            neg_mean.append(np.mean(train_neg_feature[:, i]))

    # Neg - each column std
    neg_std = []
    for i in range(8):
        if i == 2:
            neg_std.append(np.std(attribute_3_non_missing_neg_array))
        elif i == 3:
            neg_std.append(np.std(attribute_4_non_missing_neg_array))
        elif i == 5:
            neg_std.append(np.std(attribute_6_non_missing_neg_array))
        elif i == 7:
            neg_std.append(np.std(attribute_8_non_missing_neg_array))
        else:
            neg_std.append(np.std(train_neg_feature[:, i]))

    prediction = []
    # Predict
    for each_test in test_data_feature:
        pred_y1 = 0
        for i in range(8):
            if each_test[i] != 0:
                pred_y1 += np.log(1 / np.sqrt(2 * np.pi * pos_std[i] * pos_std[i])) - ((each_test[i] - pos_mean[i]) * (each_test[i] - pos_mean[i]) / (2 * pos_std[i] * pos_std[i]))
        pred_y1 += np.log(p_y1)

        pred_y0 = 0
        for i in range(8):
            if each_test[i] != 0:
                pred_y0 += np.log(1 / np.sqrt(2 * np.pi * neg_std[i] * neg_std[i])) - ((each_test[i] - neg_mean[i]) * (each_test[i] - neg_mean[i]) / (2 * neg_std[i] * neg_std[i]))
        pred_y0 += np.log(p_y0)

        if pred_y0 > pred_y1:
            prediction.append(0)
        else:
            prediction.append(1)
    
    
    correct = 0
    false = 0
    for i in range(len(prediction)):
        if test_data_label[i] == prediction[i]:
            correct += 1
        else:
            false += 1
    accuracy += correct / (correct + false)

    counter += 1

accuracy /= 10
print("Part 1B: ", accuracy)
