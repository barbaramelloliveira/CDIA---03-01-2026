from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from gerar_dataset import gerar_dataset

# 1. Gerar dados
df, X, y = gerar_dataset(n_samples=2000, seed=42)

# 2. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Previsão
y_pred = model.predict(X_test)

# 5. Avaliação
print(classification_report(y_test, y_pred))