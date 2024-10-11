Introspective Failure Prediction

Generate data set
- Put all CARLA recordings into folders called part1, part2, etc.
- Run split_into_sequences.py

Train model
- Run main.py

Evaluate and deploy model
- For testing, run main.py --eval_mode test
- For deployment, use predict.py
- predict.py takes 60 state vectors (speed, angle, throttle) as input
- Set throttle to -1 if "BRAKE" is active