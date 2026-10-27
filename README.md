AZ91D corrosion evolution surrogate

Code and numerical results accompanying the manuscript "Multiphysics simulation-based surrogate modeling of two-dimensional corrosion evolution in AZ91D magnesium alloy".

Repository contents

code/comsol contains the microstructure generators and the Java programs used to run and export the two-dimensional COMSOL reference trajectories. The three Java programs correspond to the original 108 trajectories, the second 108 trajectories, and the 144-trajectory extension.

code/machine_learning contains the U-Net implementations and the scripts used for training, validation, final evaluation, spatial-permutation analysis, and timing.

checkpoints contains the five final best checkpoints trained with 252 trajectories. manifests contains the trajectory splits and nested training subsets. results contains the numerical tables reported in the manuscript. data_index.csv lists the processed field and metric files by trajectory and split.

Data and environment

The full processed dataset contains 360 trajectories, each with 27 saved states. The final training set has 252 trajectories; validation has 36. The two independent evaluation groups contain 18 trajectories each. The processed fields total 79.65 GiB and are not stored in this Git repository. The data_index.csv file records the 720 field and metric files; a separate data deposit is required to obtain them.

The Python dependencies are listed in requirements.txt. Reproducing the reference solves requires a licensed COMSOL Multiphysics installation and the original model input. CUDA hardware is needed to reproduce the reported GPU timings. Data locations can be supplied through AZ91D_DATA_ROOT, AZ91D_PROJECT_ROOT, AZ91D_EXTENSION_ROOT, and AZ91D_FREEZE_ROOT. COMSOL paths can be supplied with the Java properties az91d.root, az91d.out, az91d.source, and az91d.caseFile. No model definition, split, seed, loss, or checkpoint should be changed when reproducing the reported results.

Reproduction sequence

1. Generate the microstructure inputs using the scripts in code/comsol.
2. Run the appropriate Java reference solver to export the trajectory fields and metrics.
3. Place the processed trajectories according to the roles in manifests and data_index.csv.
4. Set the data-location environment variables and run the training and evaluation scripts in code/machine_learning.
5. Compare the outputs with the tables in results. The five final checkpoints are provided for evaluation without retraining.

The full field dataset and the proprietary COMSOL model input are not included in this repository. Access information for the separate field-data deposit will be added when it becomes available.
