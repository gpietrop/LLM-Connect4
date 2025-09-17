# LLM-Connect4

**LLM-Connect4**: A Connect Four test-case generation framework using Large Language Models and GP agents, introduced in **“Large Language Model-based Test Case Generation for GP Agents”**, **GECCO 2024**.

If you are interested in the Mario GP experiments instead of Connect Four, please refer to [MarioGP-T](https://github.com/giorgia-nadizar/MarioGP-T).

## Contents

- `lgp_evolution_c4.py` – Runs the evolution process for LGP agents with Connect Four.
- `cgp_evolution_c4.py` – Runs the evolution process for CGP agents with Connect Four.
- `LLM_policies/` – Directory containing various policy modules generated with different LLM (e.g. `policies_31_8B.py`).
- `run_tests_*.sh` – Shell scripts to automate running experiments across policy versions.
- `results_cgp/` – Folder storing evolution results from CGP experiments.
- `results_lgp/` – Folder storing evolution results from LGP experiments.
- `analysis/` – Analysis scripts (e.g. `to_bp_tkz.py`, `bp_winning_epoch.py`, etc.).
- `testc4.py` is to provide an interface example of a game against the Random Policy.

##  Quick Start

### 1. **Clone the repository**
   ```bash
   git clone https://github.com/gpietrop/LLM-Connect4.git
   cd LLM-Connect4
   ```

### 2. **Setup Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 3. **Run Experiments**

   There is a different `.sh` script for each LLM (policy version):  

   ```bash
   chmod +x run_tests_*.sh
   ./run_tests_31_8B.sh
   ./run_tests_31_405B.sh
   ./run_tests_GPT.sh
   ```
   Each script already sets the correct policy version.
   If you want to customize the experiments (e.g., change the number of generations or the population size), just open the script and edit the variables at the top:
   ```bash
   n_generations=200   # number of generations
   n_individuals=80    # population size
   ```
   
The output files are stored in either the `results_lgp` or `results_cgp` folder, depending on the GP variant used. These folders are created automatically inside the project dierctory after running the .sh files. 
Inside each of these, the structure is organized as follows:
   - A subfolder named after the LLM used (e.g., `31_8B_NEW`).
   - Within it, subfolders follow the naming convention `results_X_Y_False`, where:
     - `X` = number of individuals
     - `Y` = number of generations
     - `False` = indicates whether automatic generation changes after a win are enabled (always set to `False` in this paper)
   - Each `results_X_Y_False` folder contains one subfolder per run (e.g., `connect4_trial_1` for run 1), storing:
     - All results for the different curricula
     - The best individual found in that run


## Run Analysis
   After experiments are complete, you can generate analysis plots using the `bp.py` script inside the `analysis/` folder. Example usage:
   ```bash
   # Run using the same arguments as in the .sh script used for running the experiments
   python analysis/bp.py --llm_model GPT --gp_model cgp --n_generations 200 --n_individuals 80

   ```
   Arguments:
   - `--llm_model` – LLM model name (default: `GPT`)
   - `--gp_model` – GP variant (default: `lgp`)
   - `--n_generations` – number of generations (default: `100`)
   - `--n_individuals` – number of individuals (default: `50`)
   - `--seed_first` – first seed (default: `1`)
   - `--seed_last` – first seed (default: `30`)


## Citation 
If you use this code, please cite: 
```
  @inproceedings{
  jorgensen2024large,
  title={Large Language Model-based Test Case Generation for GP Agents},
  author={Jorgensen, Steven and Nadizar, Giorgia and Pietropolli, Gloria and Manzoni, Luca and Medvet, Eric and O'Reilly, Una-May and Hemberg, Erik},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={914--923},
  year={2024}
}

@article{jorgensen2025policy,
  title={Policy Search through Genetic Programming and LLM-assisted Curriculum Learning},
  author={Jorgensen, Steven and Nadizar, Giorgia and Pietropolli, Gloria and Manzoni, Luca and Medvet, Eric and O'Reilly, Una-May and Hemberg, Erik},
  journal={Under review},
  year={2025}
}
```
