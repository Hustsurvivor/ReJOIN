import argparse
from main import main

if __name__ == '__main__':
    # python main.py -e 200 -g 1 -tg 4 -se 100 -s ./saved_model/group4-200/
    args = argparse.Namespace()

    args.agent_config = "config/ppo.json" # Agent configuration file
    args.network_spec = "config/complex-network.json" # Network specification file
    args.save_agent = "./saved_model/groupall-200/" # Save agent to this dir
    
    args.testing = False 
    
    args.episodes = 200 # Number of episodes
    args.save_episodes = 100 # Save agent every x episodes
    
    args.groups = 0 # Total groups of different number of relations
    args.target_group = 0 # A specific group
    args.run_all = True
    
    args.mode = "round" # Incremental Mode
    
    args.max_timesteps = 20 # Maximum number of timesteps per episode
    
    args.query = "" # Run specific query
    
    args.restore_agent = "" # Restore Agent from this dir
    args.outputs = "./outputs/" # Restore Agent from this dir
    
    args.phase = 1 # 1. cost 2. latency
    
    main(args)