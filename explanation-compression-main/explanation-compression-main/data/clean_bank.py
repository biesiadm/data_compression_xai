import pandas as pd
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("data/bank-full.csv", sep=";")

X = X.drop(["day", "month"], axis=1)

X_ = pd.concat([
    X.age,
    # pd.get_dummies(X.job, prefix='job', drop_first=True), # many values
    # pd.get_dummies(X.marital, prefix='marital', drop_first=True), # 3 values
    # pd.get_dummies(X.education, prefix='education', drop_first=True), # 4 values
    # pd.get_dummies(X.default, prefix='default', drop_first=True), # 2 values
    X.balance,
    # pd.get_dummies(X.housing, prefix='housing', drop_first=True), # 2 values
    # pd.get_dummies(X.loan, prefix='loan', drop_first=True), # 2 values
    # pd.get_dummies(X.contact, prefix='contact', drop_first=True), # 3 values
    X.duration,
    X.campaign,
    X.pdays,
    X.previous,
    # pd.get_dummies(X.poutcome, prefix="poutcome", drop_first=True) # 4 values
], axis=1)

X_ = pd.DataFrame(StandardScaler().fit_transform(X_), columns=X_.columns)

X = pd.concat([
    X_,
    pd.get_dummies(X.y, prefix="y", drop_first=True)
], axis=1)

# X.to_csv("data/bank-clean.csv", index=False)
# X.to_csv("data/bank-clean-nocat.csv", index=False)
X.to_csv("data/bank-clean-nocat-nobinary.csv", index=False)