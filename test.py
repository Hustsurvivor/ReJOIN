from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import json
import os

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from src.environment import ReJoin
import tensorflow as tf
import argparse
import logging
import sys
os.environ["KMP_AFFINITY"] = "none"
os.environ["KMP_WARNINGS"] = "0"


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # ~~~~~~~~~~~~~~~~~ Setting up the Model ~~~~~~~~~~~~~~~~~ #

    # Initialize environment (tensorforce's template)
    memory = {}
    environment = ReJoin(
        args.phase,
        args.query,
        args.episodes,
        args.groups,
        memory,
        args.mode,
        args.target_group,
        args.run_all,
        args.testing,
        args.save_path 
    )

    if args.agent_config is not None:
        with open(args.agent_config, "r") as fp:
            agent_config = json.load(fp=fp)
    else:
        raise KeyError("No agent configuration provided.")

    if args.network_spec is not None:
        with open(args.network_spec, "r") as fp:
            network_spec = json.load(fp=fp)
    else:
        raise KeyError("No network configuration provided.")

    # Set up the PPO Agent
    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states, actions=environment.actions, network=network_spec,
            variable_noise=0.5
        ),
    )

    if args.restore_agent != "":
        agent.restore_model(directory=args.restore_agent)

    runner = Runner(agent=agent, environment=environment)
    # ~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~ #

    report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:

            if args.save_agent != "" and args.testing is False and r.episode == args.save_episodes:
                save_dir = os.path.dirname(args.save_agent)
                if not os.path.isdir(save_dir):
                    try:
                        # os.mkdir(save_dir, 0o755)
                        os.makedirs(save_dir, 0o755, exist_ok=True)
                    except OSError:
                        raise OSError("Cannot save agent to dir {} ()".format(save_dir))

                    r.agent.save_model(
                        directory=args.save_agent, append_timestep=True
                    )

            logger.info(
                "Episode {ep} reward: {r}".format(ep=r.episode, r=r.episode_rewards[-1])
            )
            logger.info(
                "Average of last 100 rewards: {}\n".format(
                    sum(r.episode_rewards[-100:]) / 100
                )
            )
        return True

    logger.info(
        "Starting {agent} for Environment '{env}'".format(agent=agent, env=environment)
    )

    # Start training or testing
    runner.run(
        episodes=args.episodes,
        max_episode_timesteps=args.max_timesteps,
        episode_finished=episode_finished,
        deterministic=args.testing,
    )

    runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    
    # plt.figure(1)
    # plt.hist(runner.episode_rewards)
    #
    # plt.figure(2)
    # plt.plot(runner.episode_rewards, "b.", MarkerSize=2)

def my_test(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # ~~~~~~~~~~~~~~~~~ Setting up the Model ~~~~~~~~~~~~~~~~~ #

    # Initialize environment (tensorforce's template)
    memory = {}
    environment = ReJoin(
        args.phase,
        args.query,
        args.episodes,
        args.groups,
        memory,
        args.mode,
        args.target_group,
        args.run_all,
        args.testing,
        args.save_path
    )

    if args.agent_config is not None:
        with open(args.agent_config, "r") as fp:
            agent_config = json.load(fp=fp)
    else:
        raise KeyError("No agent configuration provided.")

    if args.network_spec is not None:
        with open(args.network_spec, "r") as fp:
            network_spec = json.load(fp=fp)
    else:
        raise KeyError("No network configuration provided.")

    # Set up the PPO Agent
    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states, actions=environment.actions, network=network_spec,
            variable_noise=0.5
        ),
    )
    
    assert args.restore_agent != ""
    agent.restore_model(directory=args.restore_agent)
    # ~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~ #


    logger.info(
        "Starting {agent} for Environment '{env}'".format(agent=agent, env=environment)
    )
    
    print('start testing!')
    while True:
        state = environment.reset()
        if state is None:
            break 
        done = False
        step = 0

        while not done:
            action = agent.act(states=state, deterministic=True)  # 确保是确定性策略
            # print(f"Step {step}, Action taken: {action}")  # 打印每一步的动作
            state, reward, done = environment.execute(action)
            step += 1
    
    print('finish testing!')


if __name__ == '__main__':
    args = argparse.Namespace()

    args.agent_config = "config/ppo.json" # Agent configuration file
    args.network_spec = "config/complex-network.json" # Network specification file
    args.save_agent = "./saved_model/groupall-200/" # Save agent to this dir
    
    args.testing = True
    args.save_path = "data/imdb-test/result.txt"
    
    args.episodes = 200 # Number of episodes
    args.save_episodes = 100 # Save agent every x episodes
    
    args.groups = 0 # Total groups of different number of relations
    args.target_group = 0 # A specific group
    args.run_all = True
    
    args.mode = "round" # Incremental Mode
    
    args.max_timesteps = 20 # Maximum number of timesteps per episode
    
    args.query = "" # Run specific query
    
    args.restore_agent = "./saved_model/groupall-200/" # Restore Agent from this dir
    args.outputs = "./outputs/" # Restore Agent from this dir
    
    args.phase = 1 # 1. cost 2. latency
    
    my_test(args)