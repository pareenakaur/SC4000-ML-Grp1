## Transactions History Model 
Key Features:
1. Purchase amounts and their statistics
2. Transaction frequency
3. Authorization patterns
4. Installment usage patterns
5. Day/time preferences
6. Category usage patterns

---

Model Evaluation:
1. Best Model: LightGBM 
2. Best Hyperparams: {'num_leaves': 153, 'max_depth': 4, 'learning_rate': 0.03261382062906748, 'n_estimators': 405, 'subsample': 0.9365415445970839, 'colsample_bytree': 0.6203360949599604, 'reg_alpha': 0.7138954369146628, 'reg_lambda': 0.5919830399874004}
3. Best RMSE: 3.715


LightGBM with Complete Dataset
1. Best RMSE is 2.806545277462748 with hyperparams: {'num_leaves': 167, 'max_depth': 12, 'learning_rate': 0.02938, 'n_estimators': 499, 'subsample': 0.9903, 'colsample_bytree': 0.7187, 'reg_alpha': 0.3956, 'reg_lambda': 8.9524}
2. Worst RMSE is 3.698463049226055 with hyperparams: {'num_leaves': 132, 'max_depth': 7, 'learning_rate': 0.011042489247777827, 'n_estimators': 414, 'subsample': 0.7426067752929947, 'colsample_bytree': 0.6880098979952209, 'reg_alpha': 1.2189617511926265e-05, 'reg_lambda': 2.2725907956937057}
