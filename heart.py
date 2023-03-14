import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

df = pd.read_csv('heart.csv')

df['target'].value_counts().plot.bar();
plt.xlabel('0: Không bị bệnh tim, 1: Bị bệnh tim')
plt.ylabel('Count');

# Nam=1 Nữ=0
print('Tổng số Nam, Nữ\n',df['sex'].value_counts())
print('Tổng số Nam, Nữ mắc bệnh\n',pd.crosstab(df['sex'], df['target']))
pd.crosstab(df['sex'], df['target']).plot(kind='bar');
plt.title('Tần suất bệnh tim theo giới tính')
plt.xlabel('0: Không bị bệnh tim, 1: Bị bệnh tim')
plt.ylabel('Count')
plt.legend(['Nữ', 'Nam']);
plt.xticks(rotation=0);

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        model_scores[model_name] = score

    return model_scores

model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)
print('Độ chính xác model: ',model_scores)

model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot(kind='bar');
plt.show()

log_reg_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}
# set up grid hyperparameter search for Logistic Regression
gs_log_reg = GridSearchCV(LogisticRegression(),log_reg_grid,cv=5,verbose=True)

# train the model
gs_log_reg.fit(X_train, y_train)
print('Thông số tốt nhất: ',gs_log_reg.best_params_)
print('Điểm: ',gs_log_reg.score(X_test, y_test))

# Evaluating Models

# make predictions
y_preds = gs_log_reg.predict(X_test)
print(classification_report(y_test, y_preds))

def predict_Heart_Disease(age,sex,cp,trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    x = np.zeros(len(X.columns))
    x[0] = age
    x[1] = sex
    x[2] = cp
    x[3] = trestbps
    x[4] = chol
    x[5] = fbs
    x[6] = restecg
    x[7] = thalach
    x[8] = exang
    x[9] = oldpeak
    x[10] = slope
    x[11] = ca
    x[12] = thal
    return gs_log_reg.predict([x])[0]

print(predict_Heart_Disease(90, 0, 0, 150, 160, 0, 0, 150,0, 3, 2, 1 ,2 ))
