# Assignment 1: Setup & Debugging Journal
**MLOps & ML Programming (2026)**

## Student Information
* **Name:** Sam Vu
* **Student ID:** 15623211
* **TA Name:** J.W.J. Hutter
* **GitHub Repository:** https://github.com/samvubs/MLOps_2026#
* **Base Skeleton Used:** [https://github.com/SURF-ML/MLOps_2026/tree/main](https://github.com/SURF-ML/MLOps_2026/tree/main)

# Question 1: First Contact with Snellius

1. **Connection Details**
   - **Command:** `ssh scur2395@snellius.surf.nl`
   - **Login Node:**  int6
   - **Screenshot:** ![Terminal Welcome Message](assets/ssh_welcome.png)

2. **Issues Encountered:**
   - **Error Message:** `The authenticity of host 'snellius.surf.nl (145.136.63.192)' can't be established.`
   - **Resolution:** 
    1. I identified that this was not a connection refused error but just a standard SSH security protocol for first-time connections.
    2. I typed `yes` to add the cluter to my `known_hosts` file.
    3. I then entered my password when prompted to complete the login.

3. **Smooth Connection (If applicable):**
   - **SSH Client:** (OpenSSH_10.0p2, LibreSSL 3.3.6)
   - **Prior Experience:** Webtech
   - **Preemptive Steps:** None that I actively took.

## Question 2: Environment Setup
1. **Setup Sequence:**
   - **Commands:** 
    12  module purge
    13  module load 2025
    14  module load Python/3.13.1-GCCcore-14.2.0
    15  module load matplotlib/3.10.3-gfbf-2025a
    16  python -m venv venv
    17  source venv/bin/activate
    18  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    19  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   - **Full Venv Path:** `/gpfs/home3/scur2395/MLOps_2026/venv'

2. **Pip Install Torch:**
   - **Duration:** Approximately 2 minutes
   - **Warnings:** None
   - **Venv Size:** `4.3G	venv`

3. **Mistakes/Unexpected Behavior:**
While the installation went well the handbook notes that using venv over Conda is a specific strategy for Snellius. I guess a mistake would be using Conda in this environment because it creates small files that perform poorly on this filesystem. By using the 2025 moduel stack and a standard venv, I ensured that the heavy Python installation is managed by the system while my specific project libraries remain isolated.

4. **Verification:**
   - **Output:** `PyTorch: 2.5.1+cu121`
`CUDA available: False`
   - **Explanation:** Because we are running this on a Login Node, which is a shared node for interactive tasks like editing code and script management. These nodes do not have GPU hardware, to see it true I will have to submit the job to a GPU-accelarated compute node.

## Question 3: Version Control Setup
1. **GitHub URL:** https://github.com/samvubs/MLOps_2026#
2. **Authentication:** SSH
3. **.gitignore:**
   - **Contents:** [__pycache__/
*.pyc
experiments/logs/
experiments/results/
.env
.DS_Store
*.egg-info/]
   - **Important items to include:** 
   * `venv/`: You must ignore this because it contains gigabytes of external libraries that are platform-dependent and should not be versioned.
   * `data/` and `*.pt`: large datasets and model weights are "binary blobs" that bloat the repository and should be stored in specialized in storage, not Git.
   * `.env`: This file often contains sensitive API keys or secrets that should never be leaked to a public or shared repository
   - **README info:** 
4. **Git Log:** `[Paste output of git log --oneline]`