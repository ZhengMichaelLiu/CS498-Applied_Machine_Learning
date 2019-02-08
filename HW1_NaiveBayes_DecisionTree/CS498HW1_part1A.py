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

    # calculate class prob
    p_y1 = len(train_pos) / len(train_data)
    p_y0 = 1 - p_y1

    # Pos - each column mean
    pos_mean = []
    for i in range(8):
        pos_mean.append(np.mean(train_pos_feature[:, i]))

    # Pos - each column std
    pos_std = []
    for i in range(8):
        pos_std.append(np.std(train_pos_feature[:, i]))

    # Neg - each column mean
    neg_mean = []
    for i in range(8):
        neg_mean.append(np.mean(train_neg_feature[:, i]))

    # Neg - each column std
    neg_std = []
    for i in range(8):
        neg_std.append(np.std(train_neg_feature[:, i]))

    prediction = []
    # Predict
    for each_test in test_data_feature:
        pred_y1 = 0
        for i in range(8):
            pred_y1 += np.log(1 / np.sqrt(2 * np.pi * pos_std[i] * pos_std[i])) - ((each_test[i] - pos_mean[i]) * (each_test[i] - pos_mean[i]) / (2 * pos_std[i] * pos_std[i]))
        pred_y1 += np.log(p_y1)

        pred_y0 = 0
        for i in range(8):
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
print(accuracy)
