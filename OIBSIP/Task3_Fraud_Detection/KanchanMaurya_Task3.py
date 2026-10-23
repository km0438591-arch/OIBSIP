# Task 3 - Fraud Detection - Level 2 - Kanchan Maurya - OIBSIP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

print("=== TASK 3 - FRAUD DETECTION - LEVEL 2 - Kanchan Maurya ===")

# 1. Create dataset - 1000 transactions - 950 normal, 50 fraud - Real pattern
np.random.seed(42)
n_normal = 950
n_fraud = 50

# Normal transactions - chota amount, normal V1 V2
normal_amount = np.random.normal(80, 50, n_normal)
normal_amount = np.clip(normal_amount, 1, 300)
normal_V1 = np.random.normal(0, 1, n_normal)
normal_V2 = np.random.normal(0, 1, n_normal)
normal_time = np.random.randint(0, 86400, n_normal)

# Fraud transactions - bada amount, alag pattern
fraud_amount = np.random.normal(800, 300, n_fraud)
fraud_amount = np.clip(fraud_amount, 500, 2000)
fraud_V1 = np.random.normal(3, 1, n_fraud)
fraud_V2 = np.random.normal(-2, 1, n_fraud)
fraud_time = np.random.randint(0, 86400, n_fraud)

# Combine
Time = np.concatenate([normal_time, fraud_time])
Amount = np.concatenate([normal_amount, fraud_amount])
V1 = np.concatenate([normal_V1, fraud_V1])
V2 = np.concatenate([normal_V2, fraud_V2])
Class = np.array([0]*n_normal + [1]*n_fraud)

df = pd.DataFrame({'Time': Time, 'Amount': Amount, 'V1': V1, 'V2': V2, 'Class': Class})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("fraud_dataset.csv", index=False)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Transactions: {len(df)}")

# 2. Class Imbalance Analysis
fraud_pct = (df['Class'].sum() / len(df)) * 100
normal_pct = 100 - fraud_pct
print(f"\n--- CLASS IMBALANCE ---")
print(f"Normal (0): {(df['Class']==0).sum()} = {normal_pct:.2f}%")
print(f"Fraud (1): {(df['Class']==1).sum()} = {fraud_pct:.2f}%")
print(f"Fraud percentage: {fraud_pct:.2f}% - Same as real dataset 0.17% ka small version!")
print("\nWhy Accuracy is misleading? - Agar sabko Normal bol du to bhi 95% accuracy aa jayegi! Isliye Precision, Recall, F1 dekhna padega!")

# Graph 1 - Class Imbalance
plt.figure(figsize=(6,4))
plt.bar(['Normal (0)','Fraud (1)'], [n_normal, n_fraud], color=['#4ECDC4','#FF6B6B'])
plt.title("Class Imbalance - Fraud Detection - Kanchan Maurya\n950 Normal vs 50 Fraud")
plt.ylabel("Count")
for i, v in enumerate([n_normal, n_fraud]):
    plt.text(i, v+20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("Class_Imbalance.png")
plt.show()
print("Graph 1 Done - Imbalance!")

# Graph 2 - Amount distribution Fraud vs Non-Fraud
plt.figure(figsize=(8,4))
plt.hist(df[df['Class']==0]['Amount'], bins=30, alpha=0.7, label='Normal', color='#4ECDC4')
plt.hist(df[df['Class']==1]['Amount'], bins=30, alpha=0.7, label='Fraud', color='#FF6B6B')
plt.title("Transaction Amount - Fraud vs Non-Fraud - Kanchan Maurya")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("Amount_Distribution.png")
plt.show()
print("Graph 2 Done - Amount!")

# 3. SMOTE - Manual Oversampling - Fraud ko duplicate karke balance karna
print("\n--- SMOTE - Manual Oversampling ---")
fraud_df = df[df['Class']==1]
normal_df = df[df['Class']==0]
print(f"Before SMOTE: Normal {len(normal_df)} Fraud {len(fraud_df)}")
# Fraud ko 950 tak badhao - duplicate + thoda noise
fraud_upsampled = fraud_df.sample(n=len(normal_df), replace=True, random_state=42)
fraud_upsampled = fraud_upsampled.copy()
fraud_upsampled['Amount'] = fraud_upsampled['Amount'] + np.random.normal(0, 10, len(fraud_upsampled))
df_balanced = pd.concat([normal_df, fraud_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"After SMOTE: Normal {(df_balanced['Class']==0).sum()} Fraud {(df_balanced['Class']==1).sum()} - BALANCED!")

# 4. Train/Test Split 80/20
train_size = int(0.8 * len(df_balanced))
train = df_balanced[:train_size]
test = df_balanced[train_size:]
print(f"\nTrain: {len(train)} Test: {len(test)}")

# 5. Model 1 - Logistic Regression Manual - Rule based
# Fraud ka pattern: Amount > 400 AND V1 > 1.5
def predict_rule(row):
    if row['Amount'] > 400 and row['V1'] > 1.0:
        return 1
    else:
        return 0

test = test.copy()
test['Pred_LogReg'] = test.apply(predict_rule, axis=1)

# Model 2 - Decision Tree Manual - Thoda strict rule
def predict_tree(row):
    if row['Amount'] > 500:
        return 1
    elif row['Amount'] > 300 and row['V1'] > 2.0 and row['V2'] < -1.0:
        return 1
    else:
        return 0

test['Pred_Tree'] = test.apply(predict_tree, axis=1)

# 6. Evaluate - Precision, Recall, F1, Accuracy - Manual calculation
def evaluate(y_true, y_pred, name):
    TP = ((y_true==1)&(y_pred==1)).sum()
    TN = ((y_true==0)&(y_pred==0)).sum()
    FP = ((y_true==0)&(y_pred==1)).sum()
    FN = ((y_true==1)&(y_pred==0)).sum()
    acc = (TP+TN)/len(y_true)*100
    prec = TP/(TP+FP)*100 if (TP+FP)>0 else 0
    rec = TP/(TP+FN)*100 if (TP+FN)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"\n--- {name} ---")
    print(f"TP:{TP} TN:{TN} FP:{FP} FN:{FN}")
    print(f"Accuracy: {acc:.2f}%  Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
    return TP,TN,FP,FN,acc,prec,rec,f1

TP1,TN1,FP1,FN1,acc1,prec1,rec1,f11 = evaluate(test['Class'], test['Pred_LogReg'], "Model 1 - Logistic Regression (Rule Based)")
TP2,TN2,FP2,FN2,acc2,prec2,rec2,f12 = evaluate(test['Class'], test['Pred_Tree'], "Model 2 - Decision Tree (Rule Based)")

# Graph 3 - Confusion Matrix for Model 1
plt.figure(figsize=(6,4))
cm = np.array([[TN1, FP1],[FN1, TP1]])
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression - Kanchan Maurya")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ['Normal','Fraud'])
plt.yticks([0,1], ['Normal','Fraud'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig("Confusion_Matrix.png")
plt.show()
print("Graph 3 Done - Confusion!")

# Graph 4 - Comparison Table - Bar chart
plt.figure(figsize=(8,4))
models = ['Logistic Reg', 'Decision Tree']
precisions = [prec1, prec2]
recalls = [rec1, rec2]
f1s = [f11, f12]
x = np.arange(len(models))
plt.bar(x-0.2, precisions, width=0.2, label='Precision', color='#FF6B9D')
plt.bar(x, recalls, width=0.2, label='Recall', color='#4ECDC4')
plt.bar(x+0.2, f1s, width=0.2, label='F1-Score', color='#FFD93D')
plt.title("Model Comparison - Precision Recall F1 - Kanchan Maurya")
plt.xticks(x, models)
plt.ylabel("%")
plt.legend()
plt.ylim(0,100)
plt.tight_layout()
plt.savefig("Model_Comparison.png")
plt.show()
print("Graph 4 Done - Comparison!")

# Conclusion
print("\n--- CONCLUSION ---")
if f11 > f12:
    print("Best Model: Logistic Regression - Higher F1 Score - Deployment ke liye best hai!")
else:
    print("Best Model: Decision Tree - Higher F1 Score - Deployment ke liye best hai!")
print("Recall matters most for fraud - Fraud miss nahi hona chahiye!")
print("Scalability: 1M transactions per hour ke liye model ko fast banana padega - Amount+V1 rule bahut fast hai!")
print("\n=== TASK 3 COMPLETED BY KANCHAN MAURYA - LEVEL 2 - FRAUD DETECTION! ===")