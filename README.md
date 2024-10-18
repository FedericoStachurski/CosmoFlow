# COSMOFlow
**Cosmological parameter inference using gravitational waves and machine learning.**  
COSMOFlow uses **normalising flows** to compute complex probability distributions for cosmological and population parameters inferred from gravitational wave events resulting from compact binary coalescences in the universe. 

### What does COSMOFlow do?
COSMOFlow is designed to compute posterior distributions of the rate of expansion of the universe, specifically the **Hubble constant (H₀)**, using posterior samples from gravitational wave events such as binary black holes. This project leverages Bayesian inference and machine learning to model these distributions efficiently.

![Flow Layout](COSMOFlow/Flow_layout.png)

### Key Features:
- **Efficient Inference**: Use trained normalising flow models to obtain Bayesian posteriors on cosmological parameters such as H₀ in seconds.
- **Scalability**: Applicable to different compact binary coalescence events (e.g., binary black holes, neutron stars) and cosmological models.
- **Use of Prior Information**: Incorporates prior knowledge from galaxy catalogues to improve parameter estimation.

## Paper
For a detailed explanation of the methodology, please refer to the paper associated with this project:  
[**Inferring Cosmological Parameters using Normalizing Flows**](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.123547), published in Physical Review D.

### Abstract:
We present a machine learning approach using normalizing flows for inferring cosmological parameters from gravitational wave events. Our methodology is general to any type of compact binary coalescence event and cosmological model and relies on the generation of training data representing distributions of gravitational wave event parameters. These parameters are conditional on the underlying cosmology and incorporate prior information from galaxy catalogues.  
We demonstrate this approach by inferring the **Hubble constant (H₀)** using binary black hole events detected during the O1, O2, and O3 observational runs of the advanced LIGO/VIRGO detectors. The resulting posterior estimate is 𝐻₀ = 74.51⁢+14.80  
−13.63 km s⁻¹ Mpc⁻¹. Our trained normalizing flow model can compute results in 𝒪(1) second.

---
## Installation

Steps for Git and Environment Setup

1. Clone the Repository
   To get a local copy of the repository:
   - Open your terminal or command prompt.
   - Run the following command to clone the repository:
   
     git clone https://github.com/your-username/your-repo.git

     Replace 'your-username/your-repo' with the actual URL of the repository you are cloning (this is the URL of the GitHub repository).

2. Navigate into the Repository
   Once the repository is cloned, navigate into the project directory:

     cd your-repo

3. Create a New Branch
   To create a new branch and switch to it, use the following commands:

     git checkout -b new-branch-name

     Replace 'new-branch-name' with the desired name for the branch. This will both create and switch to the new branch.

4. Install the Virtual Environment Using Conda
   If the project contains a 'environment.yml' file (or similarly named YAML file), you can set up the virtual environment using Conda:
   
   - Make sure you have Conda installed. You can check by running:

     conda --version

     If it's not installed, you can install it from here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html.

   - To create and activate the environment from the YAML file:

     conda env create -f environment.yml

     Replace 'environment.yml' with the actual name of the YAML file if it's different. This will install all the dependencies listed in the file.

   - Once the environment is created, activate it:

     conda activate env-name

     Replace 'env-name' with the name of the environment as defined in the 'environment.yml' file.

6. Running the Code
   Once the environment is set up and activated, you can run the project-specific scripts. For example, to run a Python script:


   Generating training data:

     python3 make_events_MLP_Galaxy_catalog_H0_v2.py -Name H0_galaxy_catalogue_ -in_out 1 -type training -mass_distribution PowerLaw+Peak -name_pop BBH -zmax 5.0 -zmin 0.00001 -H0 70 -H0max 140 -H0min 20 -SNRth 11.0 -SNRth_single 0.0 -band K -run O3 -N 1000 -Nselect 3 -device cuda:0  -threads 10 -NSIDE 32 -batch 1 -seed 114201 -detectors H1 L1 V1 -save_timer 1 -approximator IMRPhenomXPHM

   
   Training the NF:

      python3 train_flow_new_v1.py -galaxy_catalog True -data_path run_O3_det_H1_L1_V1_name_H0_galaxy_catalogue__catalog_True_band_K_batch_{}_N_100000_SNR_11_Nelect_3_Full_para_v1 -Name Flow_O3_H1_L1_V1_14target_1Cond -batch 50000 -train_size 0.8 -flow_type CouplingNSF -epochs 200 -linear_transform lu -log_it 0  -neurons 32 -layers 3 -nblock 2 -n_cond 1 -lr 0.01 -device cuda:0 -xyz 0 -Scaler MinMax -save_step 1000 -batches_data 1 -lr_scheduler CosineAnnealingLR --Volume_preserving False -n_inputs 14

   
   Make H0 posteriors:

     python3 get_all_posteriors_new_v1.py --Flow Flow_O3_H1_L1_V1_14target_1Cond --population BBH --run O3 -det HLV --device cpu --SNRth 11 --samples 1000 --Name_folder O3_HLV

Summary Steps:
1. Clone the repo: git clone https://github.com/your-username/your-repo.git
2. Navigate into the directory: cd your-repo
3. Create a new branch: git checkout -b new-branch-name
4. Install environment: conda env create -f environment.yml
5. Activate environment: conda activate env-name
6. Run your code: python script_name.py


## Authors and acknowledgment
I would like to thank Christopher Messenger, Martin Hendry, Jessica Irwin, Michael Williams.

