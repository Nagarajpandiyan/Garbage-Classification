# Garbage Classification using CNNs (Part A)

This project implements a Convolutional Neural Network (CNN) to classify images from a subset of the Garbage Classification Dataset into six categories: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`. The work includes training a CNN from scratch, performing hyperparameter sweeps, visualizing filters, and applying guided backpropagation for interpretability. 

---

## **Project Overview**

The goal of this project is to:

1. Build a CNN model from scratch with configurable parameters.
2. Train and validate the model on the Garbage Classification Dataset.
3. Explore hyperparameter tuning using **Weights & Biases (WandB)** sweeps.
4. Visualize learned features and neuron activations.
5. Evaluate the model on the test set with qualitative visualizations.

---

## **Dataset**

- The dataset contains images of garbage categorized into 6 classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- Data split:
  - **Training:** 80% of the data
  - **Validation:** 10% of the data
  - **Testing:** 10% of the data
- Transformations:
  - Training data includes data augmentation: random resized crop, flips, rotation, color jitter.
  - Validation and test data are resized to 128×128 and normalized.

---

## **Model Architecture**

The CNN model, `GarbageCNN`, has the following structure:

1. **Convolutional Blocks:**
   - 5 convolutional layers.
   - Each layer optionally followed by Batch Normalization.
   - Activation function after each conv layer (`ReLU`, `GELU`, `SiLU`, or `Mish`).
   - MaxPooling (2×2) after each conv layer.

2. **Fully Connected Layers:**
   - Dense layer (`fc1`) with configurable number of neurons (256, 512, or 1024).
   - Dropout layer (0.2 or 0.3) for regularization.
   - Output layer (`fc2`) with 6 neurons for the 6 classes.

3. **Activation Function:**
   - Configurable across all layers (conv + dense).

4. **Other Features:**
   - Trainable parameters and FLOPs are computed and logged.
   - Guided backpropagation for visualization of neuron activations.

---

## **Training Setup**

- **Optimizer:** AdamW
- **Loss Function:** CrossEntropyLoss
- **Learning Rate Scheduler:** ReduceLROnPlateau (based on validation accuracy)
- **Number of Epochs:** 30
- **Batch Size:** 32
- **Device:** GPU if available, otherwise CPU

### Hyperparameter Sweep (WandB)

The following parameters were tuned using WandB:

- Number of filters in each conv layer: `[32, 64, 128, 128, 256]` or `[64,128,256,256,512]`
- Activation functions: `ReLU`, `GELU`, `SiLU`, `Mish`
- Dropout: `0.2` or `0.3`
- Batch Normalization: `True` or `False`
- Dense layer neurons: `256`, `512`, `1024`
- Learning rate: `0.001`, `0.0005`

---

## **Evaluation & Visualization**

1. **Test Accuracy:**  
   - The model was evaluated on the held-out test set.  
   - Example test accuracy achieved: **~45%** (can be improved with more tuning and data augmentation).

2. **Sample Predictions:**
   - A **10×3 grid** shows test images alongside their predicted and true labels.

3. **First Layer Filters:**
   - Visualizations of all 32/64 filters from the first convolutional layer.

4. **Guided Backpropagation:**
   - Visualizes activations of 10 selected neurons in the last convolutional layer (CONV5).
   - Helps understand which image regions excite specific neurons.

5. **Sweep Analysis:**
   - Parallel coordinates plots show relationships between hyperparameters and validation accuracy.
   - Correlation tables help identify which hyperparameters impact performance.

---

## **Code Structure**

- `GarbageCNN` class – CNN model definition with flexible parameters.
- `count_parameters()` – Computes trainable parameters.
- `estimate_flops()` – Estimates FLOPs for the network.
- `GuidedBackprop` class – Generates gradients for neuron visualization.
- `train_sweep()` – Trains the model, logs metrics to WandB, evaluates test set, visualizes results.
- `visualize_filters_and_guided_backprop()` – Visualizes first-layer filters and guided backprop activations.
- WandB sweep setup and analysis for hyperparameter tuning.

---

## **Results**

- Hyperparameter sweeps allow experimentation with different architectures, activations, and learning rates.
- Observed trends:
  - Increasing filters generally improves validation accuracy.
  - Batch Normalization stabilizes training and can improve performance.
  - Activation function choice (ReLU, GELU, SiLU, Mish) affects convergence speed and final accuracy.
- Visualizations provide insight into what the model has learned.

---

## **Next Steps / Improvements**

- Increase dataset size or use pre-trained embeddings to improve test accuracy (~45% → 70–75%).
- Further hyperparameter tuning with more sweep runs.
- Implement advanced architectures or regularization techniques.
- Part B: Fine-tune pre-trained CNNs (ResNet, EfficientNet) for better performance.

---

## **How to Run**

1. Install dependencies:

```bash
pip install torch torchvision wandb matplotlib pandas
wandb.login(key='YOUR_WANDB_API_KEY')
DATA_PATH = '/path/to/Garbage classification'

# Garbage Classification – Part B: Fine-tuning a Pre-trained CNN

## **1. Objective**

The goal of Part B is to **fine-tune a pre-trained CNN model** on the Garbage Classification dataset and compare its performance with a model trained from scratch. Fine-tuning allows leveraging knowledge learned from large datasets (like ImageNet) to improve accuracy and reduce training time.

---

## **2. Dataset**

* Dataset: [Garbage Classification Dataset](https://drive.google.com/file/d/1nmqD6P14FvoMqqmIseqkuIe2Y40iM2SG/view)
* Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
* Data split:

  * 80% for training
  * 20% for validation and testing (equally split)

---

## **3. Pre-trained Model**

* Model used: **ResNet50** (from `torchvision.models`)
* Adjustments:

  * Input images resized to `224×224` to match ImageNet input size.
  * Last fully connected layer replaced with **6 neurons** for the garbage classes.

---

## **4. Fine-tuning Strategies**

Three strategies were explored:

1. **Strategy 1:** Freeze all convolutional layers and train only the last fully connected layer. 
2. **Strategy 2:** Freeze first few layers (initial blocks) and fine-tune remaining layers. 
3. **Strategy 3:** Fine-tune all layers with a smaller learning rate for pre-trained layers. 

---

## **5. Training Details**

* Optimizer: `Adam`
* Loss function: `CrossEntropyLoss`
* Batch size: 32
* Learning rate: 0.001 (last layer), 0.0001 (frozen layers for partial fine-tuning)
* Number of epochs: 10–15
* Logging: **Weights & Biases (WandB)** for hyperparameters, training/validation loss, and accuracy.

---

## **6. Evaluation**

* Model checkpoint saved as `best_model.pth`. 
* Test accuracy reported using the held-out test set. 
* **10×3 grid of sample images** from the test set showing predicted vs true labels. 

---

## **7. Observations**

* Fine-tuning significantly improves accuracy compared to training from scratch.
* Freezing initial layers reduces training time while retaining good performance.
* WandB logs allow visualizing the impact of different fine-tuning strategies.

---

## **8. Optional / Additional Tasks Completed**

* Model checkpointing for later use. 
* Comparison with scratch-trained CNN model. 
* Hyperparameter tracking and metrics visualization using WandB. 

---

## **9. How to Run**

1. Clone the repository:

```bash
git clone https://github.com/<user-id>/garbage_classification_partB.git
cd garbage_classification_partB
```

2. Install dependencies:

```bash
pip install torch torchvision matplotlib pandas wandb
```

3. Run the notebook:

```bash
jupyter notebook
```

4. Train the model or load the checkpoint `best_model.pth`.
5. View results and WandB dashboard for training metrics.

---

## **10. Results**

* Test Accuracy: 
* Sample predictions: shown in 10×3 grid.
* Model checkpoint and WandB logs saved for reproducibility.

---

## **11. File Structure**

```
garbage_classification_partB/
├─ notebooks/
│   └─ partB_finetuning.ipynb
├─ best_model.pth
├─ requirements.txt
├─ README.md
└─ wandb/   # Optional: WandB logs
```

---



