#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

if [[ -n "$1" ]]; then
    end=$1
else
    end=50
fi
total_scenes=$end
tracks=1
tracks_num=$tracks
#=====================selection Variables=================================



export PROJECT_PATH=/home/baiting/Desktop/dynamic_simplex/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA
export utils_route=/home/baiting/Desktop/dynamic_simplex/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/utils
export ROUTES_FOLDER=/data/weather
export DATA_FOLDER=/data/performance
export RECORD_FOLDER=/data/recorded
export SAVE_PATH=/home/baiting/Desktop/dynamic_simplex/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/data/expert # path for saving episodes while evaluating
export STATE_DATA=/home/baiting/Desktop/dynamic_simplex/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/data/state
export Scenarion_Description_Files=${PROJECT_PATH}/scene_generator
#==========================================================================

export CARLA_ROOT=/home/baiting/CARLA_0.9.15
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

#====================Constants Variables ==================================
export CARLA_ROOT=/home/baiting/CARLA_0.9.11
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
#export PYTHONPATH=$PYTHONPATH:${utils_route}/transfuser
export PYTHONPATH=$PYTHONPATH:${utils_route}
export LEADERBOARD_ROOT=${PROJECT_PATH}/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=4545 # same as the carla server port
export TM_PORT=3434 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/validation_routes/routes_town05_DIY.xml
export CHECKPOINT_ENDPOINT=results/sample_result.json # results file
export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
export TEAM_AGENT=${PROJECT_PATH}/leaderboard/team_code/image_agent.py
export TEAM_CONFIG=${PROJECT_PATH}/trained_models/Learning_by_cheating/model.ckpt
export Failure=0    #0 indicates non-failure; 1 indicates permanent failure; 2 indicates intemittent failure
export Configuration=GS    #DS: Dynamic Simplex; GS: Greedy Simplex; SA: Simplex Architecture; LBC:Performant Controller; AP: Safety Controller
#export RESUME=True
#============================================================================

#Initialize carla


$CARLA_ROOT/CarlaUE4.sh -quality-level=Epic -world-port=$PORT -resx=400 -resy=300 -opengl &
PID=$!
echo "Carla PID=$PID"
sleep 10


for (( j=0; j<=$end-1; j++ ))
  do
    i=$j
    m=$j
    x=1
    while [ $x -le $tracks ]
    do
        l=$x
        k=$x
        n=$x
        python3.7 scene_generator/interpreter.py \
        --project_path=$PROJECT_PATH\
        --simulation_num=${i}\
        --scene_num=${l}\
        --total_scenes=${total_scenes}\
        --routes_folder=$ROUTES_FOLDER\
        --data_folder=$DATA_FOLDER\
        --sdl=$Scenarion_Description_Files\
        --tracks=$tracks_num

        CUDA_VISIBLE_DEVICES=1 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
        --scenarios=${SCENARIOS}  \
        --routes=${ROUTES} \
        --repetitions=${REPETITIONS} \
        --track=${CHALLENGE_TRACK_CODENAME} \
        --checkpoint=${CHECKPOINT_ENDPOINT} \
        --agent=${TEAM_AGENT} \
        --agent-config=${TEAM_CONFIG} \
        --debug=${DEBUG_CHALLENGE} \
        --record=${PROJECT_PATH} \
        --resume=${RESUME} \
        --port=${PORT} \
        --trafficManagerPort=${TM_PORT} \
        --simulation_number=${j}\
        --scene_number=${k}\
        --project_path=$PROJECT_PATH\
        --routes_folder=$ROUTES_FOLDER\
        --data_folder=$DATA_FOLDER\
        --record_folder=$RECORD_FOLDER\
        --state_folder=$STATE_DATA\
        --sdl=$Scenarion_Description_Files\
        --tracks=$tracks\
        --failure_type=${Failure}\
        --configuration=$Configuration
        x=$(( $x + 1 ))
    done

done
echo "Done CARLA Simulations"
#pkill -f "CarlaUE4"
