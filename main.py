from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from src.environment import ReJoin
import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging
import sys

# import time
import os
import json


def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--agent-config",
        default="config/ppo.json",
        help="Agent configuration file",
    )
    parser.add_argument(
        "-n",
        "--network-spec",
        default="config/complex-network.json",
        help="Network specification file",
    )
    parser.add_argument(
        "-e", "--episodes", type=int, default=800, help="Number of episodes"
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=int,
        default=1,
        help="Total groups of different number of relations",
    )
    parser.add_argument(
        "-tg", "--target_group", type=int, default=4, help="A specific group"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="round", help="Incremental Mode"
    )
    parser.add_argument(
        "-ti",
        "--max-timesteps",
        type=int,
        default=20,
        help="Maximum number of timesteps per episode",
    )
    parser.add_argument("-q", "--query", default="", help="Run specific query")
    parser.add_argument("-s", "--save_agent", default="", help="Save agent to this dir")
    parser.add_argument("-r", "--restore_agent", default="", help="Restore Agent from this dir")
    parser.add_argument("-o", "--outputs", default="./outputs/", help="Restore Agent from this dir")

    parser.add_argument(
        "-t",
        "--testing",
        action="store_true",
        default=False,
        help="Test agent without learning.",
    )
    parser.add_argument('-all', '--run_all', action='store_true', default=False, help="Order queries by relations_num")
    parser.add_argument(
        "-se",
        "--save-episodes",
        type=int,
        default=100,
        help="Save agent every x episodes",
    )
    parser.add_argument("-p", "--phase", help="Select phase (1 or 2)", default=1)

    return parser.parse_args()


def print_config(args):
    print("Running with the following configuration")
    arg_map = vars(args)
    for key in arg_map:
        print("\t", key, "->", arg_map[key])


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
        args.run_all
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

    def find_convergence(eps):
        last = eps[-1]
        for i in range(1, len(eps)):
            if eps[i * -1] != last:
                print("Converged at episode:", len(eps) - i + 2)
                return True

    if find_convergence(runner.episode_rewards):
        print('收敛了!')
    else:
        print('未收敛')

if __name__ == "__main__":
    args = make_args_parser()
    print_config(args)
    main(args)
