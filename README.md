# Enhancing-Neural-Network-Robustness-using-Hybrid-Adversarial-Training

This repository contains all the code scripts used in the dissertation for evaluating the robustness of neural networks against adversarial attacks. The experiments involve training and testing a ResNet-18 model on the CIFAR-10 dataset using three different methodologies: 

1. **Experiment 1 (v1)** - Model trained solely on normal, unperturbed data.
2. **Experiment 2 (v2)** - Model trained exclusively on adversarial data.
3. **Experiment 3 (v3)** - Model trained using a hybrid approach of both normal and adversarial data.

![proj_flow](https://github.com/user-attachments/assets/9a551109-d3f7-4d6b-af8f-72e3449b5996)

   
The results of these experiments highlight the trade-offs between accuracy on clean data and robustness against adversarial attacks such as Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

## Repository Structure

```bash
├── experiment_v1
│   ├── train_resnet_18_v1.m       # Training script for Experiment 1
│   ├── test_normal_v1.m           # Testing script on normal data (v1)
│   ├── test_fgsm_v1.m             # Testing script against FGSM adversarial data (v1)
│   ├── test_pgd_v1.m              # Testing script against PGD adversarial data (v1)
├── experiment_v2
│   ├── train_resnet_18_v2.m       # Training script for Experiment 2
│   ├── test_normal_v2.m           # Testing script on normal data (v2)
│   ├── test_fgsm_v2.m             # Testing script against FGSM adversarial data (v2)
│   ├── test_pgd_v2.m              # Testing script against PGD adversarial data (v2)
├── experiment_v3
│   ├── train_resnet_18_v3.m       # Training script for Experiment 3
│   ├── test_normal_v3.m           # Testing script on normal data (v3)
│   ├── test_fgsm_v3.m             # Testing script against FGSM adversarial data (v3)
│   ├── test_pgd_v3.m              # Testing script against PGD adversarial data (v3)
└── README.md                      # This README file
```

## Results Overview

- **Experiment 1 (v1)**: 
    - Validation Accuracy on Normal Data: **91.58%**
    - Validation Accuracy under FGSM Attack: Varies between **61.31%** and **16.84%** depending on the strength of \(\epsilon\).
    - Validation Accuracy under PGD Attack: **0.57%**
    
- **Experiment 2 (v2)**: 
    - Validation Accuracy on Normal Data: **85.81%**
    - Validation Accuracy under FGSM Attack: Varies between **76.00%** and **46.01%**.
    - Validation Accuracy under PGD Attack: **38.75%**

- **Experiment 3 (v3)**:
    - Validation Accuracy on Normal Data: **88.75%**
    - Validation Accuracy under FGSM Attack: Varies between **77.30%** and **43.78%**.
    - Validation Accuracy under PGD Attack: **35.66%**

## How to Run the Scripts

### Requirements

To run these scripts, you need MATLAB R2022a or later installed, along with the following MATLAB toolboxes:
- Deep Learning Toolbox
- Parallel Computing Toolbox (for utilizing GPUs)

### Training the Models

To train the models for each experiment, navigate to the corresponding folder and run the training script. For example, to train the model for Experiment 1 (v1), use:

```matlab
cd experiment_v1
train_resnet_18_v1.m
```

This script will train the ResNet-18 model on the CIFAR-10 dataset. The trained model will be saved as `resnet_18_v1.mat`.

### Testing the Models on Normal Data

To test the model on normal, unperturbed CIFAR-10 data, use the following script after training:

```matlab
test_normal_v1.m
```

This will output the validation accuracy on clean data for the model.

### Testing the Models under Adversarial Attacks

1. **FGSM Testing**: To test the model against adversarial data generated using FGSM, run:

    ```matlab
    test_fgsm_v1.m
    ```

2. **PGD Testing**: To test the model against adversarial data generated using PGD, run:

    ```matlab
    test_pgd_v1.m
    ```

Replace `v1` with `v2` or `v3` for the other experiments. These scripts will calculate and display the model's accuracy under different adversarial perturbation strengths.

### Customizing the Adversarial Attacks

You can modify the strength of the adversarial perturbations (\(\epsilon\)) in the testing scripts. For FGSM, the value of \(\epsilon\) can be adjusted directly in the code:

```matlab
epsilon = 8; % Set the perturbation strength for FGSM
```

Similarly, for PGD, the number of iterations and the step size can be adjusted:

```matlab
epsilon = 8;  % Maximum allowed perturbation
alpha = 0.01; % Step size for each iteration
num_iterations = 40; % Number of iterations for PGD
```

## Contact

For any questions or issues with the code, please feel free to reach out via the Issues section of this repository.
