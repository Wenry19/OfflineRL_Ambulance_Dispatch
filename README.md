# Offline Reinforcement Learning for Ambulance Dispatch

Enric Lamarca Ferrés, Master's Thesis, Master's degree in Artificial Intelligence.

Universitat Politècnica de Catalunya (UPC).

Haute Ecole d'Ingénierie et de Gestion du Canton de Vaud (HEIG-VD).

**Please note that the data was not included for confidentiality reasons.**

**CHUV/**

    It contains all the CHUV data in csv files. (NOT INCLUDED)

**algorithms/**

    It contains all the implemented training algorithms.

**auxiliars/**

    It contains auxiliar scripts.

**build_experiences/**

    It contains all the code in charge of building experiences using the CHUV data.
    Later, these experiences will be used to train agents in an offline setting.

**configs/**

    It contains the hyperparameter configuration of each training execution.

**data_analysis/**

    It contains the scripts in charge of doing the data analysis of the datasets.

**environment/**

    It contains the implementation of the environment used to test and evaluate the trained agents.

**generated_data/**

    It contains the generated data in pickle format. (NOT INCLUDED)

**logs/**

    It contains different logs of the training executions (batch loss, mean reward...)
    to visualize them in tensorboard.

**models/**

    It contains the trained models.

**outs/**

    It contains auxiliar txt files. (NOT INCLUDED)

**test/**

    It contains the code in charge of testing the trained agents.

**training/**

    It contains the code in charge of training agents.

**utils.py**

    It contains useful functions.
