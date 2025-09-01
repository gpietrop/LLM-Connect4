# LLM-Connect4

**LLM-Connect4**: A Connect Four test-case generation framework using Large Language Models and GP agents, introduced in **“Large Language Model-based Test Case Generation for GP Agents”**, **GECCO 2024**.

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