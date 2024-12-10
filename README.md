# SafeDiffuseTraj

## Overview
This project investigates whether reachability criteria can make diffusion trajectory models provably safe, providing a framework for evaluation.

## Repository Contents

### Core Files
- **Diffusion_Trajectory_Planner.ipynb**: Demonstrates cases where diffusion models may fail in trajectory optimization and where Reachability techniques may succeed
- **diffmod.ipynb**: Main implementation notebook
- **diffusionsde.py**: Standard SDE implementation for diffusion model
- **hjsafediffusersde.py**: Diffuser with Reachability implementation
- **planeworldtrail.py**: Working PlaneWorld manual game

### Experimental Files
- **confused.py**: Trial implementation of plane-world with gradient reward (unused in final project)

## Requirements

### Computing Resources
- Colab Pro (for compute power and terminal access) OR
- RIT GPU (Note: Resource allocation issues were encountered)

### Dependencies
- MuJoco 210 (not 200)
- Modified CleanDiffuser framework (from [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser))

### Installation Steps

1. Set up conda environment
2. Clone and install required repositories:
```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
conda run -n cleandiffuser pip install -e .

git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
conda run -n cleandiffuser pip install -e .

git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
conda run -n cleandiffuser pip install -e .
```

## Running the Project

### Training
```bash
python pipelines/diffuser_d4rl_antmaze.py --mode train
```

### Inference
```bash
python pipelines/diffuser_d4rl_antmaze.py --mode inference
```

## Notes for RC Implementation

RC environments have specific limitations:
- No SUDO access
- Package conflicts with on-board installations
- Requires Apptainer container system environment
- Must use fakeroot feature (see [Apptainer documentation](https://apptainer.org/docs/admin/1.0/user_namespace.html#fakeroot-feature))

## Additional Resources
For further documentation and questions, refer to the [CleanDiffuserTeam documentation](https://github.com/CleanDiffuserTeam/CleanDiffuser).
