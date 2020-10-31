from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import CQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from sklearn.model_selection import train_test_split

# get data-driven RL dataset
dataset, env = get_pybullet('hopper-bullet-mixed-v0')

# split dataset
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# setup algorithm
cql = CQL(actor_learning_rate=1e-3,
          critic_learning_rate=1e-3,
          temp_learning_rate=1e-3,
          alpha_learning_rate=1e-3,
          n_critics=10,
          bootstrap=True,
          update_actor_interval=2,
          q_func_type='qr',
          use_gpu=True)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=300,
        scorers={
            'environment': evaluate_on_environment(env),
            'advantage': discounted_sum_of_advantage_scorer
        })