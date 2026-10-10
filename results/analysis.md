# AI4003 Assignment 02 — Results Analysis

## 1. Model comparison

Four models were evaluated on the same held-out test set: logistic regression using Set R, logistic regression using Set Q, a linear SVM, and an RBF SVM.

The test accuracies were 72.64%, 97.00%, 73.59%, and 99.58%, respectively. The RBF SVM achieved the highest test accuracy, followed by logistic regression using Set Q.

## 2. Effect of feature engineering

Set R contains the six original electrical measurements: Ia, Ib, Ic, Va, Vb, and Vc. Set Q adds the square of each measurement, giving twelve features.

Logistic regression accuracy increased from 72.64% with Set R to 97.00% with Set Q. This suggests that the squared features help the model represent relationships that are not adequately captured by the original features alone.

## 3. Linear versus nonlinear SVM

The linear SVM achieved 73.59% test accuracy, while the RBF SVM achieved 99.58%. A linear SVM uses a linear decision boundary, whereas an RBF kernel can represent more complex nonlinear boundaries. The observed difference suggests that the RBF kernel is better suited to separating these samples.

## 4. Confusion matrix and missed faults

The RBF SVM produced 1,301 true negatives, 4 false positives, 6 false negatives, and 1,090 true positives.

A false negative represents a fault sample classified as normal. The RBF SVM missed 6 of the 1,096 actual faults, corresponding to approximately 99.45% fault recall. In comparison, Set R logistic regression missed 657 faults, Set Q logistic regression missed 72, and the linear SVM missed 634.

The RBF SVM therefore provided the best balance of fault detection and false alarms among the evaluated models.

## 5. Limitations and deployment considerations

The dataset was randomly split into training, validation, and test subsets. If samples from nearby time periods are similar, a random split may produce optimistic performance estimates. A chronological evaluation on a separate period would provide a useful additional test of generalization.

The results also show that overall accuracy alone is not enough for fault detection. False negatives and false positives have different consequences and should both be monitored. Further validation on independent data would be necessary before considering deployment in an actual power system.

## 6. Conclusion

The RBF SVM achieved the highest test accuracy at 99.58% and missed only 6 fault samples. Logistic regression with squared features also performed well, achieving 97.00% accuracy. The results support the use of nonlinear decision boundaries for this dataset, while further independent validation remains necessary.
