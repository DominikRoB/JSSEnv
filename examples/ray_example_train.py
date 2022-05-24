import ray
from ray.rllib.utils import check_env
from ray import tune
from gym import envs
import gym
# import JSSEnv
from JSSEnv import utils
import os
from ray.tune.stopper import MaximumIterationStopper


def env_creator(env_config):
    import JSSEnv
    env = gym.make('JSSEnv-v1', env_config=env_config)
    return env


if __name__ == '__main__':
    # from ray import tune
    # from ray.tune.registry import register_env

    ## Init ray
    ray.shutdown()
    ray.init()

    ## Specify Config
    algorithm_name = "PPO"
    env_name = "JSSEnv-v1"
    evaluation_interval = 20
    evaluation_duration = 20
    evaluation_duration_unit = "episodes"
    checkpoint_freq = 2
    num_gpus = 0
    tune_local_dir = "Experiments_JSSEnv"
    instance_path = "\\instances\\ta20"
    instance_path = r"C:\MYDOCUMENTS\Repos\Promotion_Bleidorn\JSSEnv\JSSEnv\envs\instances\ta01"
    max_iterations = 1000

    env_config = {"instance_path": instance_path}

    config_dict = {"env": env_name,
                   "env_config": env_config,
                   "evaluation_interval": evaluation_interval,
                   "evaluation_duration": evaluation_duration,
                   "evaluation_duration_unit": evaluation_duration_unit,
                   "num_gpus": num_gpus,  # int(os.environ.get("RLLIB_NUM_GPUS", "0"))
                   # other configurations, if none, then default will be used
                   }

    # Create env, check and register to ray
    env = env_creator(env_config)
    check_env(env)
    tune.registry.register_env(env_name, env_creator)

    # Run Experiment

    file_path = os.path.realpath(__file__)
    print(f"Call 'tensorboard --logdir \"{os.path.join(file_path, tune_local_dir, algorithm_name)}\"'")

    stopper = MaximumIterationStopper(max_iterations)
    tune_analysis = tune.run(algorithm_name,
                             config=config_dict,
                             local_dir=tune_local_dir,
                             checkpoint_freq=checkpoint_freq,
                             stop=stopper,
                             )
    best_trial = tune_analysis.get_best_trial()
    best_checkpoint = tune_analysis.get_best_checkpoint(best_trial, mode="max")

    from ray.rllib.agents.ppo.ppo import PPOTrainer

    trained_agent = PPOTrainer(config=config_dict)
    trained_agent.restore(best_checkpoint.local_path)

    obs = env.reset()
    explore = False
    while True:
        action = trained_agent.compute_single_action(obs, explore=explore)
        if not env.legal_actions[action]:
            explore = True
            continue

        obs, reward, done, _ = env.step(action)
        if env.legal_actions[action]:
            env.render()
        if len(env.legal_actions) == 0:
            print("Deadlock, no legal actions")
            break
        if done:
            print("Episode ended")
            break
    env.close()
    ray.shutdown()
