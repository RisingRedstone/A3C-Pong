# A3C-Pong

To train type:
python3 train.py --n_workers {No of workers, default : 1, recommended : 16} PongNoFrameskip-v4

To test an existing one type:
python3 train.py --testing True PongNoFrameskip-v4 --savedir {Saved Directory + /network}

Example:
python3 train.py --testing True PongNoFrameskip-v4 --savedir Saves/Save2/network

To save a network while training, press 's'

This is a very raw code, will refine it and provide explaination of how it works later. :)
