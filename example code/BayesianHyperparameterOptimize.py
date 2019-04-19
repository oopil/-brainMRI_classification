from bayes_opt import BayesianOptimization
import numpy as np
def target(x):
    return np.exp(-(x-3)**2) + np.exp(-(3*x-2)**2) + 1/(x**2+1)
bayes_optimizer = BayesianOptimization(target, {'x': (-2, 6)},
                                       random_state=0)
bayes_optimizer.maximize(init_points=2, n_iter=14, acq='ei', xi=0.01)


""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

# 중간 점검
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. 학습 수행 및 성능 평가를 위한 기본 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # 평균 이미지
np.save('/tmp/asirra_mean.npy', image_mean)    # 평균 이미지를 저장
hp_d['image_mean'] = image_mean

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 200

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: 정규화 관련 하이퍼파라미터
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: 성능 평가 관련 하이퍼파라미터
hp_d['score_threshold'] = 1e-4


""" 3. 특정한 초기 학습률 및 L2 정규화 계수 하에서 학습을 수행한 후, 검증 성능을 출력하는 목적 함수 정의 """
def train_and_validate(init_learning_rate_log, weight_decay_log):
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    hp_d['init_learning_rate'] = 10**init_learning_rate_log
    hp_d['weight_decay'] = 10**weight_decay_log

    model = ConvNet([227, 227, 3], 2, **hp_d)
    evaluator = Evaluator()
    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

    sess = tf.Session(graph=graph, config=config)
    train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)

    # 검증 정확도의 최댓값을 목적 함수의 출력값으로 반환
    best_val_score = np.max(train_results['eval_scores'])

    return best_val_score


""" 4. BayesianOptimization 객체 생성, 실행 및 최종 결과 출력 """
bayes_optimizer = BayesianOptimization(
    f=train_and_validate,
    pbounds={
        'init_learning_rate_log': (-5, -1),    # FIXME
        'weight_decay_log': (-5, -1)            # FIXME
    },
    random_state=0,
    verbose=2
)

bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)    # FIXME

for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)
