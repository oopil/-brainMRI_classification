import time
from bayes_opt import BayesianOptimization

def BayesOptimize(self, init_lr_log, w_stddev_log):
    self.set_lr(10 ** init_lr_log)
    self.set_weight_stddev(10 ** w_stddev_log)
    print('-' * 100)
    # print('learning rate : {}\nstddev of weight : {}'.\
    #       format(self.learning_rate, 10**w_stddev_log))
    print('learning rate : {}\nstddev of weight : {}'. \
          format(self.learning_rate, 10 ** w_stddev_log))

    bayes_optimizer = BayesianOptimization(
        f=self.train(),
        pbounds={
            # 78 <= -1.2892029132535314,-1.2185073691640054
            # 85 <= -1.2254855784556566, -1.142561108840614}}
            'init_lr_log': (-2.0, -1.0),
            'w_stddev_log': (-2.0, -1.0)
        },
        random_state=0,
        # verbose=2
    )
    bayes_optimizer.maximize(
        init_points=5,
        n_iter=40,
        acq='ei',
        xi=0.01
    )
    BO_results = []
    BO_results.append('\n\t\t<<< class option : {} >>>\n'.format(self.class_option))
    BO_result_file_name = "BO_result/BayesOpt_results" \
                          + str(time.time()) + '_' + self.class_option
    fd = open(BO_result_file_name, 'a+t')
    for i, ressult in enumerate(bayes_optimizer.res):
        BO_results.append('Iteration {}:{}\n'.format(i, ressult))
        print('Iteration {}: {}'.format(i, ressult))
        fd.writelines('Iteration {}:{}\n'.format(i, ressult))
    BO_results.append('Final result: {}\n'.format(bayes_optimizer.max))
    fd.writelines('Final result: {}\n'.format(bayes_optimizer.max))
    print('Final result: {}\n'.format(bayes_optimizer.max))
    fd.close()