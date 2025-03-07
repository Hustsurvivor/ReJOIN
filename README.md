# ReJOIN
Refactored version of Rejoin

* [Rejoin: Hands-Free Query Optimizer through Deep Learning - Ryan Marcus & Olga Papaemmanouil](https://www.cs.brandeis.edu/~olga/publications/HandsFreeCIDR19.pdf)
* [Deep Reinforcement Learning for Joining Enumaration - Ryan Marcus & Olga Papaemmanouil](https://www.cs.brandeis.edu/~olga/publications/ReJOIN_aiDM18.pdf)

## set up 

```python
python create_training_file.py
```

```python
python train.py
```

```python
python test.py
```

## Program parameters

- Agent configuration file  
    "-a", "--agent-config"  
    default="config/ppo.json"  

- Network specification file  
    "-n", "--network-spec"  
    default="config/complex-network.json"  

- Number of episodes  
    "-e", "--episodes"  
    default=800  

- Total groups of different number of relations  
    "-g", "--groups"  
    default=1  

- Run specific group  
    "-tg", "--target_group"  
    default=5  

- Incremental Mode  
    "-m", "--mode"  
    default="round"  

- Maximum number of timesteps per episode  
    "-ti", "--max-timesteps"  
    default=20  

- Run specific query  
    "-q", "--query"  
     default=""  

- Save agent to this dir  
    "-s", "--save_agent"  

- Restore Agent from this dir  
    "-r", "--restore_agent"  

- Test agent without learning (use deterministic) 
    "-t", "--testing"  
    action="store_true"  
    default=False  

- Order queries by relations_num  
    "-all", "--run_all"  
    default=False  

- Save agent every x episodes  
    "-se", "--save-episodes"  
    default=100  

- Select phase (1 or 2)  
    "-p", "--phase"  
    default=1  
    
