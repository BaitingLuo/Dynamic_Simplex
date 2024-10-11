# Dynamic Simplex: Balancing Safety and Performance in Autonomous Cyber Physical Systems
This repository contains the implementation for [Dynamic Simplex: Balancing Safety and Performance in Autonomous Cyber Physical Systems](https://dl.acm.org/doi/pdf/10.1145/3576841.3585934) (ICCPS 2023).

**Abstract**:  
> Learning Enabled Components (LEC) have greatly assisted cyberphysical systems in achieving higher levels of autonomy. However,
LEC’s susceptibility to dynamic and uncertain operating conditions
is a critical challenge for the safety of these systems. Redundant controller architectures have been widely adopted for safety assurance
in such contexts. These architectures augment LEC “performant”
controllers that are difficult to verify with “safety” controllers and
the decision logic to switch between them. While these architectures ensure safety, we point out two limitations. First, they are
trained offline to learn a conservative policy of always selecting
a controller that maintains the system’s safety, which limits the
system’s adaptability to dynamic and non-stationary environments.
Second, they do not support reverse switching from the safety
controller to the performant controller, even when the threat to
safety is no longer present. To address these limitations, we propose
a dynamic simplex strategy with an online controller switching
logic that allows two-way switching. We consider switching as a
sequential decision-making problem and model it as a semi-Markov
decision process. We leverage a combination of a myopic selector
using surrogate models (for the forward switch) and a non-myopic
planner (for the reverse switch) to balance safety and performance.
We evaluate this approach using an autonomous vehicle case study
in the CARLA simulator using different driving conditions, locations, and component failures. We show that the proposed approach
results in fewer collisions and higher performance than state-ofthe-art alternatives.

Follow the setup instructions below to run dynamic simplex in CARLA simulator. The simulator and the decision logic can be run inside a docker. Follow these instructions to build and launch the docker.

# Create Folders for Docker Volumes
Create three folders named ```routes```, ```simulation-data``` and ```recorded``` inside the data directory. These folders are the data volumes for the carla client docker. Run the following

```
mkdir data
mkdir data/routes               #stores the scene information.
mkdir data/simulation-data      #stores the sensor information
mkdir data/recorded             #stores the sensor data chosen by the user
```
Alternately, enter into this repo and execute this script ```./make_volume_folders.sh``` to set up these empty folders.

# Downloads

***Step 1***: Manual Download: Download [CARLA_0.9.10](https://github.com/carla-simulator/carla/releases/tag/0.9.10/) and put it inside the AV-Adaptive-Mitigation folder. Please pull CARLA_0.9.10 version. If you pull any other version, there will be a mismatch between the CARLA simulation API and the client API. 

Automated Downloads (Preferred): Enter into this repo and execute this script ```./pull_carla_simulator.sh``` to download these three requirements automatically into the required folders.

***Step 2***: The preitrained controllers can be got from the following repos:

1. [Learning By Cheating](https://github.com/bradyz/2020_CARLA_challenge). Unzip the weights file and save it as ***model.ckpt*** in the trained_models/Learning_by_cheating folder. 

# Docker Build

***Step 1***: Make sure NVIDIA Docker is set up. Please follow these [steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide) if you have not already set it up.

***Step 2***: Run the following to build a docker using Ubuntu 18.04 image and then install all the libraries required to run the our carla client

```
./build_carla_client.sh
```
Please make sure the build is complete without errors. ```You will get sudo pip warning and that should be fine```

***Step 3***: After the docker is built. Set up xhost permission for the GUI to show up the simulation run. For this open a terminal on your local machine and run the following command 

```
xhost -local:docker
```
***Note***: Possible erros "Cannot connect to X server: 1" . The solution is to run **xhost +** in a terminal on your host computer.  [Reference](Reference: https://stackoverflow.com/questions/56931649/docker-cannot-connect-to-x-server)



# Docker Run

Now, run the carla client docker using the script below. 

```
./run_carla_client.sh
```  
This will take you into the carla client docker. When you are inside the docker run the following

```
./run_evaluation.sh
```

***Note***: If running shows cuda image kernel issue, you may need to execute the command below inside docker. (Due to the performance model used, this repository currently does not support CUDA 12 or above.)
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

This image_agent.py script includes all critical codes of getting control actions from safety controllers, performant controllers.
```
/ANTI-CARLA/leaderboard/team_code/image_agent.py
```

This MCTS_random.py script includes all critical codes of performing our MCTS algorithm.
```
/ANTI-CARLA/leaderboard/team_code/MCTS_random.py
```

All results will be saved under ```/ANTI-CARLA/data```

Inside ```./run_evaluation.sh```, there are a few variables that need to be set before execution.
1. end=1 => Change the number 1 to any number to indicate the times of running simulations. Each new simulation will start with a new initial condition under random_weather folder.
2. PORT => The simulator port (default:2000), Carla simulator is defaultly set to 2000.
3. Failure =>  0 indicates non-failure; 1 indicates permanent failure; 2 indicates intemittent failure
4. Configuration => DS: Dynamic Simplex; GS: Greedy Simplex; SA: Simplex Architecture; LBC:Performant Controller; AP: Safety Controller

# Citation
```
@inproceedings{10.1145/3576841.3585934,
author = {Luo, Baiting and Ramakrishna, Shreyas and Pettet, Ava and Kuhn, Christopher and Karsai, Gabor and Mukhopadhyay, Ayan},
title = {Dynamic Simplex: Balancing Safety and Performance in Autonomous Cyber Physical Systems},
year = {2023},
isbn = {9798400700361},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3576841.3585934},
doi = {10.1145/3576841.3585934},
abstract = {Learning Enabled Components (LEC) have greatly assisted cyber-physical systems in achieving higher levels of autonomy. However, LEC's susceptibility to dynamic and uncertain operating conditions is a critical challenge for the safety of these systems. Redundant controller architectures have been widely adopted for safety assurance in such contexts. These architectures augment LEC "performant" controllers that are difficult to verify with "safety" controllers and the decision logic to switch between them. While these architectures ensure safety, we point out two limitations. First, they are trained offline to learn a conservative policy of always selecting a controller that maintains the system's safety, which limits the system's adaptability to dynamic and non-stationary environments. Second, they do not support reverse switching from the safety controller to the performant controller, even when the threat to safety is no longer present. To address these limitations, we propose a dynamic simplex strategy with an online controller switching logic that allows two-way switching. We consider switching as a sequential decision-making problem and model it as a semi-Markov decision process. We leverage a combination of a myopic selector using surrogate models (for the forward switch) and a non-myopic planner (for the reverse switch) to balance safety and performance. We evaluate this approach using an autonomous vehicle case study in the CARLA simulator using different driving conditions, locations, and component failures. We show that the proposed approach results in fewer collisions and higher performance than state-of-the-art alternatives.},
booktitle = {Proceedings of the ACM/IEEE 14th International Conference on Cyber-Physical Systems (with CPS-IoT Week 2023)},
pages = {177–186},
numpages = {10},
location = {San Antonio, TX, USA},
series = {ICCPS '23}
}
```