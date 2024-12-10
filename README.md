# SafeDiffuseTraj
Can the use of reachability criteria make diffusion trajectory models provably safe? We provide a framework by which to evaluate this. 

Files contain the following:
Diffusion_Trajectory_Planner.ipynb
A lowlevel program to show an example where diffusion models may fail in trajectory optimization, and where Reachability techniques may succeed. 

confused.py
A trial implementation of the plane-world with gradient reward. Unused in final project. 

diffmod.ipynb
Requirements - Colab Pro for compute power and terminal access, or RIT GPU. Was unable to use RIT GPU due to resource allocation issues. Modification of the CleanDiffuser paradigm (https://github.com/CleanDiffuserTeam/CleanDiffuser). However, their packages require several extraneous installations. If trying to reproduce, include MuJoco 210 (not 200). If implementing on RC, note the following:

RC does not allow SUDO access, which hinders installation ability, as there exist conflicts between requirements and on-board packages.
Therefore, to install, one must create a fake-root in Apptainer, RIT's supported container system environment. This requires some energy. Refer to https://apptainer.org/docs/admin/1.0/user_namespace.html#fakeroot-feature for details.
Outputs show the successful training for the models. To run, use conda installer, and the following git clones:

!git clone https://github.com/Farama-Foundation/D4RL.git; cd D4RL; conda run -n cleandiffuser pip install -e .
!git clone https://github.com/ARISE-Initiative/robomimic.git; cd robomimic; conda run -n cleandiffuser pip install -e .
!git clone https://github.com/ARISE-Initiative/robosuite.git; cd robosuite; conda run -n cleandiffuser pip install -e .

For training, run python pipelines/diffuser_d4rl_antmaze.py (mode=train). For inference, use (mode=inference).
Refer to CleanDiffuserTeam documentation for further questions. 

diffusionsde.py
Standard SDE for implementation of diffusion model.
hjdiffusersde.py: 
diffuser with Reachability implementation

planeworldtrail.py
Working PlaneWorld manual game.
