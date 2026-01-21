# Assignment 2: MLOps & PCAM Pipeline Journal
**MLOps & ML Programming (2026)**

## Group Information
* **Group Number:** [Your Group Number]
* **Team Members:** [Member 1 Name/ID, Member 2 Name/ID, Member 3 Name/ID, Member 4 Name/ID, Member 5 Name/ID]
* **GitHub Repository:** [Link to your Group Repository]
* **Base Setup Chosen from Assignment 1:** [Name of the group member whose repo was used as the foundation]

---

## Question 1: Reproducibility Audit
1. **Sources of Non-Determinism:**
* GPU Algorithms: 
   * Non-deterministic behavior in CuDNN's convolution algorithm, which trade strict order of operations for computational speed. By default you add numbers in order you hear them but because of how computers handle decimals (floating point math), A + B + C might give a tiny bit different result than C + A + B. 
* Shuffling vs. Sampling:
   * The `WeightedRandomSampler` in `loader.py` introduces randomness by probabillistically selecting samples to balance the class dsitributions. Unlike a standard shuffle this sampler biases the selection to pick rare cases more frequently.
* Random model-initialization: 
   * Without a fixed seed the hidden layers in the MLP are initialized with  different random weights each run.
* Library seeds: 
   * Python's `random` module and `numpy` generate pseudo-random numbers which causes data processing results to vary if not seeded.

2. **Control Measures:**
* GPU Algorithms:
   * Currently Controlled? Yes
   * How is it controlled?
      * By setting `torch.backends.cudnn.deterministic = True` and `benchmark = False`, we force CuDNN to use deterministic algorithms, ensuring the same order of operations every time.
* Shuffling vs. Sampling:
   * Currently Controlled: Yes
   * How is it controlled:
      * The `WeightedRandomSampler` relies on PyTorch's rando number generator. We control this by setting the PyTorch seed (`torch.manual_seed`) in our `set_seed` function.
* Model Initialization
   * Currently Controlled: Yes
   * How is it controlled:
      * We call `torch_manual_seed()` before initializing the model. This ensures PyTorch generates the exact same starting weights for the MLP layers in every run.
* Library Seeds
   * Currently Controlled: Yes
   * How is it controlled:
      * We explicitly set the seeds for external libraries using `random.seed(seed)` and `np.random.seed(seed)` to ensure any auxiliary data processing is consistent.

3. **Code Snippets for Reproducibility:**
   ```python
   # Paste the exact code added for seeding and determinism
   def set_seed(seed): """Zet alle seeds vast voor volledige reproduceerbaarheid.""" 
      random.seed(seed) 
      np.random.seed(seed) 
      torch.manual_seed(seed) #… non-determinism op bij model initialisatie.
      torch.cuda.manual_seed(seed) 
      torch.cuda.manual_seed_all(seed)

      #… het vastzetten van de gpu gedrag, door PyTorch te dwingen om 		altijd hetzelfde pad te kiezen.

      torch.backends.cudnn.deterministic = True 
      torch.backends.cudnn.benchmark = False 
      print(f"Reproducibility: Seed ingesteld op {seed}")
   ```

4. **Twin Run Results:**

| User | Run # | Final Train Loss | Final Val Loss | Result |
| :--- | :---: | :--- | :--- | :--- |
| **Sam** | 1 | 0.3751 | 0.7829 | — |
| **Sam** | 2 | 0.007548 | 0.564752 | ✅ Match |
| **[Teammate]** | 1 | 0.007548 | 0.564752 | ✅ Match |
| **[Teammate]** | 2 | 0.007548 | 0.564752 | ✅ Match |

---

## Question 2: Data, Partitioning, and Leakage Audit
1. **Partitioning Strategy:**

2. **Leakage Prevention:**
   
3. **Cross-Validation Reflection:**

4. **The Dataset Size Mystery:**

5. **Poisoning Analysis:**

---

## Question 3: Configuration Management
1. **Centralized Parameters:**
The following five parameters were identified as critical variables that should not be hard-coded:
* Learning Rate: This controls the step of the optimizer
* Batch Size: Determines how many images are processed simultaneously.
* Epochs: The number of times the model processes the entire dataset.
* Hidden Units: Defines the model architecture of the MLP.
Data Path: Specifies the location of the PCAM files on the Snellius server.

2. **Loading Mechanism:**
   - We used a YAML-based configuration management to begin with.

   * Configuration File (`config.yaml`): The parameters are stored in a structured YAML format.
   ```python
   data:
      data_path: "/scratch-shared/scur2395/surfdrive"
      batch_size: 128
      num_workers: 2

   model:
      name: "mlp"
      input_shape: [3, 96, 96]
      num_classes: 1
      hidden_units: 128

   training:
      epochs: 5
      learning_rate: 0.001
      log_interval: 100
      save_dir: "experiments/results"
   ```
   * Loading in `train.py`: We use `yaml.safe_load` to inject these values into the script at runtime. This allows the code to adapt without manual editing:
   ```python
   def main(config_path):
      # 1. Load Config
      with open(config_path, "r") as f:
         cfg = yaml.safe_load(f)
   
      ## example of loaded parameters:
      optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
      epochs = cfg['training']['epochs']
   ```

3. **Impact Analysis:**
* Reproducibility: Isolating configuration in a seperate file allows us to track which settings produced a specific result.
* Experiment Comparison: Comparing different runs is now a matter of comparing YAML files rather than searching through source code.
* Collaboration: Teammates can run the code on  their own accounts by changing a single line in `config.yaml` instead of editing `train.py`, reducing bugs and conflicts.

4. **Remaining Risks:** 
* With configuration management there are still some risks:
* Software Environment: YAML does not track versions of libraries like TORCH or CUDA which can cause discrepancies
* Data Integrity: The config points to a path but cannot guarantee the data there has not been modified or corrupted.
* Human Error: Standard YAML files are vulnerable to mistakes like typos.

---

## Question 4: Gradients & LR Scheduler
1. **Internal Dynamics:**

2. **Learning Rate Scheduling:**

---

## Question 5: Part 1 - Experiment Tracking
1. **Metrics Choice:**

2. **Results (Average of 3 Seeds):**

3. **Logging Scalability:**

4. **Tracker Initialization:**
   ```python
   # Snippet showing tracker/MLFlow/W&B initialization
   ```

5. **Evidence of Logging:**

6. **Reproduction & Checkpoint Usage:**

7. **Deployment Issues:**

---

## Question 5: Part 2 - Hyperparameter Optimization
1. **Search Space:**
2. **Visualization:**
3. **The "Champion" Model:**

4. **Thresholding Logic:**

5. **Baseline Comparison:**

---

## Question 6: Model Slicing & Error Analysis
1. **Visual Error Patterns:**

2. **The "Slice":**

3. **Risks of Silent Failure:**

---

## Question 7: Team Collaboration and CI/CD
1. **Consolidation Strategy:** 
2. **Collaborative Flow:**

3. **CI Audit:**

4. **Merge Conflict Resolution:**

5. **Branching Discipline:**

---

## Question 8: Benchmarking Infrastructure
1. **Throughput Logic:**

2. **Throughput Table (Batch Size 1):**

| Partition | Node Type | Throughput (img/s) | Job ID |
| :--- | :--- | :--- | :--- |
| `thin_course` | CPU Only | | |
| `gpu_course` | GPU ([Type]) | | |

3. **Scaling Analysis:**

4. **Bottleneck Identification:**

---

## Question 9: Documentation & README
1. **README Link:** [Link to your Group Repo README]
2. **README Sections:** [Confirm Installation, Data Setup, Training, and Inference are present.]
3. **Offline Handover:** [List the files required on a USB stick to run the model offline.]

---

## Final Submission Checklist
- [ ] Group repository link provided?
- [ ] Best model checkpoint pushed to GitHub?
- [ ] inference.py script included and functional?
- [ ] All Slurm scripts included in the repository?
- [ ] All images use relative paths (assets/)?
- [ ] Names and IDs of all members on the first page?