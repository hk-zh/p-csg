# Penalty-based Imitation Learning with Cross Semantics Generation Sensor Fusion for Autonomous Driving

<img src="figures/architecture.png" height="256" hspace=30>

## Contents

1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Data Generation](#data-generation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Acknowledgements](#acknowledgements)

## Setup

Install anaconda

```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Build the environment

```Shell
conda create -n p-csg python=3.9
conda activate p-csg
pip3 install -r requirements.txt
cd leaderboard
pip3 install -r requirements.txt
```

Download and setup CARLA 0.9.10.1

```Shell
chmod +x setup_carla.sh
./setup_carla.sh
```

## Dataset

The data is generated with ```leaderboard/team_code/auto_pilot.py``` in 8 CARLA towns using the routes and scenarios files provided at ```leaderboard/data``` on CARLA 0.9.10.1

```Shell
chmod +x download_data.sh
./download_data.sh
```

We used two datasets for different experimental settings:

- clear_weather_data: contains only `ClearNoon` weather. This dataset is used for the experiments described in the paper and generalization to new town results shown in the [video](https://youtu.be/WxadQyQ2gMs).
- 14_weathers_data: contains 14 preset weather conditions mentioned in ```leaderboard/team_code/auto_pilot.py```. This dataset is used for training models for the [leaderboard](https://leaderboard.carla.org/leaderboard) and the generalization to new weather results shown in the [video](https://youtu.be/WxadQyQ2gMs).

The dataset is structured as follows:

```
- TownX_{tiny,short}: corresponding to different towns and routes files
    - routes_X: contains data for an individual route
        - rgb_front: multi-view camera images at 400x300 resolution
        - seg_front: corresponding segmentation images
        - lidar: 3d point cloud in .npy format
        - topdown: topdown segmentation images 
        - measurements: contains ego-agent's position, velocity and other metadata
        - stop_sign: the stop sign information used for training stop sign indicator and apply stop sign penalty
        - light: the red light information along with stop line locations before intersections. It is used for training traffic light indicator and apply red light penalty
```

## Data Generation

In addition to the dataset, we have also provided all the scripts used for generating data and these can be modified as required for different CARLA versions.

### Running CARLA Server

#### With Display

```Shell
./CarlaUE4.sh --world-port=2000 --tm-port=8000 -windowed -ResX=640 -ResY=480 -fps=15
```

#### Without Display

Without Docker:

```
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh --world-port=2000 --tm-port=8000
```

With Docker:

Instructions for setting up docker are available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Pull the docker image of CARLA 0.9.10.1 ```docker pull carlasim/carla:0.9.10.1```.

Docker 18:

```
docker run -it --rm -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 --tm-port=8000
```

Docker 19:

```Shell
docker run -it --rm --net=host --gpus '"device=0"' carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 --tm-port=8000
```

If the docker container doesn't start properly then add another environment variable ```-e SDL_AUDIODRIVER=dsp```.

### Run the Autopilot

Once the CARLA server is running, rollout the autopilot to start data generation.

```Shell
./leaderboard/scripts/run_evaluation.sh
```

The expert agent used for data generation is defined in ```leaderboard/team_code/auto_pilot.py```. Different variables which need to be set are specified in ```leaderboard/scripts/run_evaluation.sh```. The expert agent is based on the autopilot from [this codebase](https://github.com/bradyz/2020_CARLA_challenge).

### Routes and Scenarios

Each route is defined by a sequence of waypoints (and optionally a weather condition) that the agent needs to follow. Each scenario is defined by a trigger transform (location and orientation) and other actors present in that scenario (optional). The [leaderboard repository](https://github.com/carla-simulator/leaderboard/tree/master/data) provides a set of routes and scenarios files. To generate additional routes, spin up a CARLA server and follow the procedure below.

#### Generating routes with intersections

The position of traffic lights is used to localize intersections and (start_wp, end_wp) pairs are sampled in a grid centered at these points.

```Shell
python3 tools/generate_intersection_routes.py --save_file <path_of_generated_routes_file> --town <town_to_be_used>
```

#### Sampling individual junctions from a route

Each route in the provided routes file is interpolated into a dense sequence of waypoints and individual junctions are sampled from these based on change in navigational commands.

```Shell
python3 tools/sample_junctions.py --routes_file <xml_file_containing_routes> --save_file <path_of_generated_file>
```

#### Generating Scenarios

Additional scenarios are densely sampled in a grid centered at the locations from the [reference scenarios file](https://github.com/carla-simulator/leaderboard/blob/master/data/all_towns_traffic_scenarios_public.json). More scenario files can be found [here](https://github.com/carla-simulator/scenario_runner/tree/master/srunner/data).

```Shell
python3 tools/generate_scenarios.py --scenarios_file <scenarios_file_to_be_used_as_reference> --save_file <path_of_generated_json_file> --towns <town_to_be_used>
```

## Training

To train the model, please run the following code

```Shell
CUDA_VISIBLE_DEVICES=0 python3 team_code/train.py 
```

The important training arguments we have are following:

```
--id: Unique experiment identification
--device: training device to use
--epoch: the number of train epochs
--lr: learning rate
--val_every: validation frequence
--bacth_size: batch size
--logdir: the directory to log data (checkpoints, arguments information)
--lambda1: the weight of red light penalty
--lambda2: the weight of speed penalty
--lambda3: the weight of stop sign penalty
```

## Evaluation

Spin up a CARLA server (described above) and run the required agent. The adequate routes and scenarios files are provided in ```leaderboard/data``` and the required variables need to be set in ```leaderboard/scripts/run_evaluation.sh```.

```Shell
CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh  <carla root> <working directory>
```

## Acknowledgements

- This work was supported by Huawei Trustworthy Technology and Engineering Laboratory.

- This work uses code from the following open-source projects and datasets:
  - TransFuser: https://github.com/autonomousvision/transfuser (License [MIT](https://github.com/autonomousvision/transfuser/blob/2022/LICENSE))
  - Carla Leaderboard: https://github.com/carla-simulator/leaderboard (License [MIT](https://github.com/carla-simulator/leaderboard/blob/master/LICENSE))

