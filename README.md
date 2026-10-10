# Three-Phase Transmission Line Fault Detection

## Project Overview

This project investigates fault detection in a three-phase transmission line dataset using logistic regression implemented from scratch and Support Vector Machines (SVMs). It compares different feature representations and model configurations to evaluate their ability to classify normal and faulty conditions.

This project was completed for the AI4003 Applied Machine Learning course.

## Dataset

The dataset file is `data/detect_dataset.csv`.

The six input features are:
- Ia
- Ib
- Ic
- Va
- Vb
- Vc

The target is `Output (S)`, representing the binary class label.

Completely empty columns are removed during data preparation. The data is divided into training, validation, and test sets using the assignment's required random permutation and seed.

## Methods

### Logistic Regression

Logistic regression is implemented manually using NumPy. The implementation includes the sigmoid function, binary cross-entropy cost, batch gradient descent, and probability-based classification.

Two feature sets are compared:

- **Set R:** Six original features, standardized using training-set statistics.
- **Set Q:** Six original features plus their squares, standardized using training-set statistics.

Several learning rates are evaluated, and the selected rate is used to train the final models.

### Support Vector Machines

SVM models are trained using `sklearn.svm.SVC`, as permitted by the assignment.

- Linear kernel: different values of C are compared.
- RBF kernel: different values of C and gamma are compared.

Model selection uses validation accuracy. The final selected models are evaluated on the test set.

## Data Split

The dataset is split into:
- 60% training
- 20% validation
- Remaining rows for testing

The random generator seed is 42. Feature scaling parameters are calculated using only the training data.

## Results

The recorded test results are:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression — Set R | 72.64% |
| Logistic Regression — Set Q | 97.00% |
| Linear SVM | 73.59% |
| RBF SVM | 99.58% |

The RBF SVM achieved the highest test accuracy among the evaluated models. Its test confusion matrix recorded six false negatives (missed faults) and four false positives.

These results apply to this dataset and experimental setup. They do not establish that the model is ready for real-world transmission-line protection.

See `results/analysis.md` and `results/test_results.csv` for further analysis and recorded metrics.

## Project Structure

```text
AI4003_Assignment02/
├── app/
│   └── app.py
├── data/
│   └── detect_dataset.csv
├── src/
│   ├── load_data.py
│   ├── prepare.py
│   ├── eda.py
│   ├── logistic.py
│   ├── test_logistic_functions.py
│   ├── train_logistic.py
│   ├── train_svm.py
│   └── test_models.py
└── results/
    ├── analysis.md
    ├── prepared_data.npz
    ├── learning_rate_results.csv
    ├── svm_validation_results.csv
    ├── test_results.csv
    ├── figures/
    ├── models/
    └── screenshots/
```

## Installation

Install the required packages in your Python environment:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn joblib streamlit
```

## Running the Project

Run these commands from the main project folder.

Prepare the dataset:

```powershell
python .\src\prepare.py
```

Generate exploratory data analysis figures:

```powershell
python .\src\eda.py
```

Train the logistic regression models:

```powershell
python .\src\train_logistic.py
```

Train and compare SVM models:

```powershell
python .\src\train_svm.py
```

Evaluate the final models:

```powershell
python .\src\test_models.py
```

Launch the demonstration frontend:

```powershell
python -m streamlit run .\app\app.py
```

## Limitations

The dataset-based experiment does not establish performance on live transmission lines. A random split may also produce optimistic estimates if observations are related or collected close together in time. Real deployment would require independent testing, protection-system engineering, and safety validation.
