# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

source activate env_MTMC



cd ..
chmod +x main.py
python main.py --ConfigPath "config/config.yaml"
cd scripts/

cd ..
chmod +x main.py
python main.py --ConfigPath "config/config2.yaml"
cd scripts/

cd ..
chmod +x main.py
python main.py --ConfigPath "config/config3.yaml"
cd scripts/








