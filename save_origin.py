def main_logistic_regression():
    '''
        set the data option and load dataset
    '''
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    # "clinic" or "new" or "PET"
    # 'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
    # diag_type = "PET"
    # class_option = 'PET pos vs neg'
    diag_type = "new"
    class_option = 'NC vs ADD'#'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    # diag_type = "clinic"
    # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    class_split = class_option.split('vs')
    class_num = len(class_split)
    excel_option = 'merge'  # P V T merge
    test_num = 20
    fold_num = 5
    is_split_by_num = False # split the dataset by fold.
    # sampling_option = 'SMOTENC'
    # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
    sampling_option_str = 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'# ADASYN
    sampling_option_split = sampling_option_str.split(' ')

    whole_set = NN_dataloader(diag_type, class_option, \
                              excel_path, excel_option, test_num, fold_num, is_split_by_num)
    whole_set = np.array(whole_set)

    result_file_name = \
    '/home/sp/PycharmProjects/brainMRI_classification/regression_result/chosun_MRI_excel_logistic_regression_result_'\
    +diag_type +'_'+ class_option
    #if there is space in the file name, i can't use it in the linux command.
    is_remove_result_file = True
    if is_remove_result_file:
        # command = 'rm {}'.format(result_file_name)
        # print(command)
        subprocess.call(['rm',result_file_name])
        # os.system(command)
    # assert False
    line_length = 100

    total_test_accur = []
    for sampling_option in sampling_option_split:
        results = []
        test_accur_list = []
        results.append('\n\t\t<<< class option : {} / oversample : {} >>>\n'.format(class_option, sampling_option))
        date = str(datetime.datetime.now())+'\n'
        results.append(date)
        # assert False
        print(len(whole_set))
        for fold_index, one_fold_set in enumerate(whole_set):
            train_num, test_num = len(one_fold_set[0]), len(one_fold_set[2])
            contents = []
            contents.append(
                'fold : {}/{:<3},'.format(fold_index, fold_num))
            line, test_accur = logistic_regression(one_fold_set, sampling_option, class_num)
            contents.append(line)
            test_accur_list.append(test_accur)
            results.append(contents)

        test_accur_avg = int(sum(test_accur_list)/len(test_accur_list))
        results.append('{} : {}\n'.format('avg test accur',test_accur_avg))
        results.append('=' * line_length + '\n')
        total_test_accur.append(test_accur_avg)

        file = open(result_file_name, 'a+t')
        # print('<< results >>')
        for result in results:
            file.writelines(result)
            # print(result)
        # print(contents)
        file.close()
    print_result_file(result_file_name)
    print(total_test_accur)

def save():
    is_merge = True  # True
    option_num = 0  # P V T options
    '''
    I should set the class options like
    NC vs AD
    NC vs MCI
    MCI vs AD

    NC vs MCI vs AD
    '''
    class_option = ['NC vs AD', 'NC vs MCI', 'MCI vs AD', 'NC vs MCI vs AD']
    class_option_index = 0
    class_num = class_option_index // 3 + 2
    # SMOTEENN and SMOTETomek is not good in FCN
    sampling_option = 'SMOTE'  # None ADASYN SMOTE SMOTEENN SMOTETomek

    ford_num = 5
    ford_index = 0
    keep_prob = 0.9  # 0.9

    learning_rate = 0.05
    epochs = 2000
    print_freq = 200
    save_freq = 200
    '''
    Log
    1. batch normalization seems to have no positive effect on validation.
    2. i need to try attention mechanism here
    3. before that, i need to find the simplest model first.
    '''
    # batch_size = 50
    data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)

    # assert False
    data, label = shuffle_two_arrays(data, label)
    X_train, Y_train, X_test, Y_test = split_train_test(data, label, ford_num, ford_index)
    # print(len(data[0]), len(X_train[0]))
    X_train, Y_train = over_sampling(X_train, Y_train, sampling_option)
    X_test, Y_test = valence_class(X_test, Y_test, class_num)
    train_num = len(Y_train)
    test_num = len(Y_test)
    feature_num = len(X_train[0])
    print(X_train.shape, X_test.shape)
    # assert False

    # In[36]:

    graph = tf.Graph()
    with graph.as_default():
        # Tensorflow placeholders - inputs to the TF graph
        inputs = tf.placeholder(tf.float32, [None, feature_num], name='Inputs')
        targets = tf.placeholder(tf.float32, [None, class_num], name='Targets')

        layer = [512, 1024, 2048, 1024]
        layer_last = 256
        with tf.name_scope("FCN"):
            # defining the network
            with tf.name_scope("layer1"):
                l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
                l2 = fully_connected_layer(l1, layer[0], layer[0], keep_prob)
                l3 = fully_connected_layer(l2, layer[0], layer[1], keep_prob)
            with tf.name_scope("layer2"):
                l4 = fully_connected_layer(l3, layer[1], layer[1], keep_prob)
                l5 = fully_connected_layer(l4, layer[1], layer[1], keep_prob)
                l6 = fully_connected_layer(l5, layer[1], layer[1], keep_prob)
                l7 = fully_connected_layer(l6, layer[1], layer[1], keep_prob)
                l8 = fully_connected_layer(l7, layer[1], layer[1], keep_prob)
            with tf.name_scope("layer5"):
                l19 = fully_connected_layer(l8, layer[1], layer[1], keep_prob)
                l20 = fully_connected_layer(l19, layer[1], layer[1], keep_prob)
                l21 = fully_connected_layer(l20, layer[1], layer[1], keep_prob)
                l22 = fully_connected_layer(l21, layer[1], layer[1], keep_prob)
                l23 = fully_connected_layer(l22, layer[1], layer[1], keep_prob)
                l24 = fully_connected_layer(l23, layer[1], layer_last, keep_prob)
                l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

            # defining special parameter for our predictions - later used for testing
            predictions = tf.nn.sigmoid(l_fin)

        # Mean_squared_error function and optimizer choice - Classical Gradient Descent
        cost = loss2 = tf.reduce_mean(tf.squared_difference(targets, predictions))
        tf.summary.scalar("cost", cost)
        merged_summary = tf.summary.merge_all()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Starting session for the graph
        top_train_accur = 0
        top_test_accur = 0
        train_accur_list = []
        test_accur_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())
            # writer = tf.train.SummaryWriter('./my_graph', graph=tf.get_default_graph())
            writer.add_graph(sess.graph)
            # TRAINING PORTION OF THE SESSION
            # one hot encoding
            '''
            search for tf.one_hot
            '''
            Y_train = pd.get_dummies(Y_train)
            Y_train = np.array(Y_train)
            Y_test = pd.get_dummies(Y_test)
            Y_test = np.array(Y_test)
            for i in range(epochs):
                '''
                idx = np.random.choice(len(X_train), batch_size, replace=True)
                x_batch = X_train[idx, :]
                y_batch = Y_train[idx]
                y_batch = np.reshape(y_batch, (len(y_batch), 1))
                '''
                y_batch = Y_train
                x_batch = X_train

                summary, batch_loss, opt, preds_train = sess.run([merged_summary, cost, optimizer, predictions],
                                                                 feed_dict={inputs: x_batch, targets: y_batch})
                writer.add_summary(summary, global_step=i)
                train_accur = accuracy(preds_train, Y_train)
                # TESTING PORTION OF THE SESSION
                preds = sess.run([predictions], feed_dict={inputs: X_test})
                # preds_nparray = np.squeeze(np.array(preds), 0)
                preds_nparray = np.squeeze(np.array(preds), 0)
                test_accur = accuracy(preds_nparray, Y_test)

                if i % save_freq == 0:
                    train_accur_list.append(train_accur // 1)
                    test_accur_list.append(test_accur // 1)

                if i % print_freq == 0:
                    # if i > (epochs//2):
                    if i >= (epochs // 2) and top_train_accur < train_accur:
                        top_train_accur = train_accur
                    if i >= (epochs // 2) and top_test_accur < test_accur:
                        top_test_accur = test_accur
                        print(top_test_accur)
                    print('=' * 50)
                    print('epoch                : ', i, '/', epochs)
                    print('batch loss           : ', batch_loss)
                    print("Training Accuracy (%): ", train_accur)
                    print("Test Accuracy     (%): ", test_accur)
                    print('pred                 :', np.transpose(np.argmax(preds_nparray, 1)))
                    print('label                :', np.transpose(np.argmax(Y_test, 1)))

            writer.close()
            print('<< top accuracy >>')
            print('Training : ', top_train_accur)
            print('Testing  : ', top_test_accur)

            for i in range(len(train_accur_list)):
                print(train_accur_list[i], test_accur_list[i])

    # In[37]:

    assert False
    import os
    line_length = 100
    # is_remove_result_file = True
    is_remove_result_file = False
    result_file_name = '/home/sp/PycharmProjects/chosun_AD/chosun_MRI_excel_AD_classification_result'
    if is_remove_result_file:
        os.system('rm {}'.format(result_file_name))
    contents = []
    contents.append('=' * line_length + '\n')
    contents.append(
        'class option : {:30} ford index / num : {}/{:<10} train and test number : {:10} / {:<10} oversample : {}\n'.format(
            class_option[class_option_index], ford_index, ford_num, train_num, test_num,
            sampling_option) + 'keep probability : {:<30} epoch : {:<30} learning rate : {:<30}\n'.format(keep_prob,
                                                                                                          epochs,
                                                                                                          learning_rate))
    contents.append('top Train : {:<10} {}\n'.format(top_train_accur // 1, train_accur_list)
                    + 'top Test  : {:<10} {}\n' \
                    .format(top_test_accur // 1, test_accur_list))

    file = open(result_file_name, 'a+t')
    file.writelines(contents)
    # print(contents)
    file.close()

    '''
    top_train_accur = 0
    top_test_accur = 0
    train_accur_list = []
    test_accur_list = []
    '''

    # In[38]:

    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

    # In[39]:

    print(1)