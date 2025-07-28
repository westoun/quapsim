# QuaPSim: Qua(ntum Simulation for) P(opulations made) Sim(ple)

This repository contains the source code to the paper
_"Cache-based Simulation for faster Quantum Circuit Synthesis and Compilation"_
by Stein, Klikovits, and Wimmer.

## Installation

You can install quapsim locally by cloning the code into your current
directory, creating a [virtual environment](https://docs.python.org/3/library/venv.html), and running

```
pip install -e .
```

## Run the evaluation

If you wish to reproduce the evaluation presented in the paper, begin by
building the docker image from the current folder via

```
docker build . -t quapsim
```

Then, navigate into the folder corresponding to the evaluation setup you
want to reproduce, either `evaluation/rcs_experiments` or `evaluation/ga_case_study`.

Within that folder, make sure the docker compose file contains the experiment
configurations you wish to execute. Then, run these experiments by calling

```
docker compose up
```

The generated experiment log are stored in the `results`-directory of each
folder. To extract data from the logs, execute the `extract_results.py` script.
The formatted data are written to a JSON file in the same folder.

Based on this JSON file, the `evaluation_*_results.ipynb` notebooks create
the images and numbers reported in the paper.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
