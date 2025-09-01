# LLM-Connect4

**LLM-Connect4**: A Connect Four test-case generation framework using Large Language Models and GP agents, introduced in **“Large Language Model-based Test Case Generation for GP Agents”**, **GECCO 2024**.


## Contents

- `lgp_evolution_c4.py` – Runs the evolution process for LGP agents with Connect Four.
- `cgp_evolution_c4.py` – Runs the evolution process for CGP agents with Connect Four.
- `LLM_policies/` – Directory containing various policy modules generated with different LLM (e.g. `policies_31_8B.py`).
- `run_tests_*.sh` – Shell scripts to automate running experiments across policy versions.
- `results_cgp/` – Folder storing evolution results from CGP experiments.
- `results_lgp/` – Folder storing evolution results from LGP experiments.
- `analysis/` – Analysis scripts (e.g. `to_bp_tkz.py`, `bp_winning_epoch.py`, etc.).

##  Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/gpietrop/LLM-Connect4.git
   cd LLM-Connect4
   ```

2. **Setup Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Experiments**

   ```bash
   chmod +x run_tests_*.sh
   ./run_tests_31_8B.sh
   ./run_tests_31_405B.sh
   ./run_tests_GPT.sh
   ```

4. The output files will be stored in either the `results_lgp` or `results_cgp` folder (depending on the GP variant used). Inside, each LLM has its own directory, with subdirectories named after individual runs. 

## Citation 
If you use this code please cite: 
```
      @inproceedings{
  jorgensen2024large,
  title={Large Language Model-based Test Case Generation for GP Agents},
  author={Jorgensen, Steven and Nadizar, Giorgia and Pietropolli, Gloria and Manzoni, Luca and Medvet, Eric and O'Reilly, Una-May and Hemberg, Erik},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={914--923},
  year={2024}
}
```