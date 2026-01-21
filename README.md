# АНАЛИЗ ДАННЫХ - ШПАРГАЛКА К ЭКЗАМЕНУ
**Дата подготовки:** 21 января 2026 | **Экзамен:** 23 января 2026

---

## ТЕМА 1: ТИПЫ ДАННЫХ И РАСПРЕДЕЛЕНИЯ

### 🔑 Ключевые понятия

**Случайная величина** - переменная, значение которой зависит от случайных факторов
- **Дискретная** (случайное количество событий)
- **Непрерывная** (значение в диапазоне)

**Детерминированная величина** - всегда имеет одно значение

### Основные распределения

| Распределение | Формула | Применение | Код |
|---|---|---|---|
| **Нормальное (Гаусса)** | $\mu, \sigma$ | Природные явления, тестовые баллы | `np.random.normal(mean, std, size)` |
| **Пуассона** | $\lambda$ | Количество событий в времени | `np.random.poisson(lam, size)` |
| **Биномиальное** | $n, p$ | Успех/неудача испытаний | `np.random.binomial(n, p, size)` |
| **Экспоненциальное** | $\lambda$ | Время до события | `np.random.exponential(scale, size)` |
| **Равномерное** | $a, b$ | Случайное число в диапазоне | `np.random.uniform(a, b, size)` |

### Характеристики распределений

```python
import numpy as np
from scipy import stats

data = np.random.normal(100, 15, 1000)

# Основные характеристики
print(f"Среднее (mean): {np.mean(data)}")
print(f"Медиана (median): {np.median(data)}")
print(f"Мода (mode): {stats.mode(data)}")
print(f"Стандартное отклонение: {np.std(data)}")
print(f"Дисперсия (variance): {np.var(data)}")
print(f"Асимметрия (skewness): {stats.skew(data)}")  # 0 = симметрично
print(f"Эксцесс (kurtosis): {stats.kurtosis(data)}")  # мера остроты пика

# Квартили
print(f"Q1: {np.percentile(data, 25)}, Q3: {np.percentile(data, 75)}")
```

### Стационарность времени

**Стационарная серия:**
- Среднее const (не изменяется со временем)
- Дисперсия const
- Автокорреляция зависит только от лага, не от времени

**Нестационарная серия:**
- Тренд (среднее растет/падает)
- Сезонность
- Изменяющаяся дисперсия

### Превращение нестационарной в стационарную

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Тест на стационарность (ADF тест)
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Критические значения: {result[4]}')
    
    if result[1] <= 0.05:
        print("✓ Серия стационарна (p < 0.05)")
    else:
        print("✗ Серия нестационарна (p > 0.05)")
    return result[1] <= 0.05

# МЕТОДЫ ПРЕВРАЩЕНИЯ:

# 1. Differencing (разности)
ts_diff = data.diff().dropna()

# 2. Log-transform
ts_log = np.log(data)

# 3. Деление на тренд
from scipy.signal import detrend
ts_detrended = detrend(data)

# 4. Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data, model='additive', period=12)
ts_deseasonalized = data - decomposition.seasonal
```

---

## ТЕМА 2: СВЯЗИ МЕЖДУ ПЕРЕМЕННЫМИ

### 📊 Числовые ↔ Числовые

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt

x = np.random.randn(100)
y = 2*x + np.random.randn(100)

# КОРРЕЛЯЦИЯ ПИРСОНА (линейная связь, параметрический метод)
corr_pearson, p_value = pearsonr(x, y)
print(f"Корреляция Пирсона: {corr_pearson:.3f}, p={p_value:.4f}")

# КОРРЕЛЯЦИЯ СПИРМЕНА (ранговая, монотонная связь)
corr_spearman, p_value = spearmanr(x, y)
print(f"Корреляция Спирмена: {corr_spearman:.3f}, p={p_value:.4f}")

# КОРРЕЛЯЦИЯ КЕНДАЛЛА (для малых выборок)
corr_kendall, p_value = kendalltau(x, y)
print(f"Корреляция Кендалла: {corr_kendall:.3f}, p={p_value:.4f}")

# Корреляционная матрица
import pandas as pd
df = pd.DataFrame({'X': x, 'Y': y})
corr_matrix = df.corr(method='pearson')  # или 'spearman'
print(corr_matrix)
```

### 📊 Категориальные ↔ Категориальные

```python
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

# Таблица сопряженности (contingency table)
data = pd.DataFrame({
    'Gender': ['M', 'M', 'F', 'F', 'M', 'F'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B']
})

contingency_table = pd.crosstab(data['Gender'], data['Product'])
print(contingency_table)
#          A  B
# Gender      
# F        2  1
# M        2  1

# CHI-SQUARE тест
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square: {chi2:.4f}, p-value: {p_value:.4f}")

# CRAMÉR'S V (нормализованная мера связи 0-1)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

v = cramers_v(data['Gender'], data['Product'])
print(f"Cramér's V: {v:.3f}")  # 0 = нет связи, 1 = полная связь
```

### 📊 Числовая ↔ Категориальная

```python
from scipy.stats import f_oneway, pointbiserialr
import pandas as pd

# Point-biserial (только 2 категории)
# Числовая переменная vs бинарная категориальная
x = np.random.randn(100)
category = np.random.randint(0, 2, 100)  # 0 или 1

corr_pb, p_value = pointbiserialr(category, x)
print(f"Point-biserial корреляция: {corr_pb:.3f}, p={p_value:.4f}")

# ANOVA (несколько групп)
df = pd.DataFrame({
    'value': np.random.randn(300),
    'group': np.repeat(['A', 'B', 'C'], 100)
})

groupA = df[df['group'] == 'A']['value']
groupB = df[df['group'] == 'B']['value']
groupC = df[df['group'] == 'C']['value']

f_stat, p_value = f_oneway(groupA, groupB, groupC)
print(f"F-статистика: {f_stat:.4f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("✓ Значимые различия между группами")
else:
    print("✗ Различия не значимы")
```

---

## ТЕМА 3: ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ И P-VALUE

### Доверительный интервал (CI)

```python
import numpy as np
from scipy import stats

data = np.random.normal(100, 15, 100)

# Доверительный интервал для среднего (95%)
n = len(data)
mean = np.mean(data)
se = stats.sem(data)  # Standard Error

# Метод 1: t-распределение
t_critical = stats.t.ppf(0.975, n-1)  # 0.975 для двустороннего теста
ci_lower = mean - t_critical * se
ci_upper = mean + t_critical * se
print(f"CI 95% (t-dist): [{ci_lower:.2f}, {ci_upper:.2f}]")

# Метод 2: z-распределение (для больших выборок)
z_critical = stats.norm.ppf(0.975)
ci_lower_z = mean - z_critical * se
ci_upper_z = mean + z_critical * se
print(f"CI 95% (z-dist): [{ci_lower_z:.2f}, {ci_upper_z:.2f}]")

# Для доли (proportion)
successes = 56
n_total = 80
p_hat = successes / n_total
se_p = np.sqrt(p_hat * (1 - p_hat) / n_total)
ci_p_lower = p_hat - 1.96 * se_p
ci_p_upper = p_hat + 1.96 * se_p
print(f"CI для доли 95%: [{ci_p_lower:.3f}, {ci_p_upper:.3f}]")
```

### P-value

```python
from scipy.stats import ttest_1samp, norm

# P-value - вероятность получить такие же или более экстремальные результаты,
# если нулевая гипотеза верна

# Пример: одновыборочный t-тест
data = np.random.normal(102, 15, 100)
null_mean = 100

t_stat, p_value = ttest_1samp(data, null_mean)
print(f"t-статистика: {t_stat:.4f}, p-value: {p_value:.4f}")

# Интерпретация:
if p_value < 0.001:
    print("✓✓✓ Очень сильные доказательства ПРОТИВ H0 (p < 0.001)")
elif p_value < 0.01:
    print("✓✓ Сильные доказательства ПРОТИВ H0 (p < 0.01)")
elif p_value < 0.05:
    print("✓ Умеренные доказательства ПРОТИВ H0 (p < 0.05)")
else:
    print("✗ Недостаточно доказательств ПРОТИВ H0 (p ≥ 0.05)")

# Связь между CI и p-value
# Если CI не содержит значение из H0 → p < 0.05
```

---

## ТЕМА 4: ПРОВЕРКА ГИПОТЕЗ (Параметрические методы)

### Проверка среднего

```python
from scipy.stats import ttest_1samp, ttest_ind
import numpy as np

# ОДНОВЫБОРОЧНЫЙ T-ТЕСТ
# H0: μ = 100
# Ha: μ ≠ 100

data = np.random.normal(102, 15, 50)
null_mean = 100

t_stat, p_value = ttest_1samp(data, null_mean)
print(f"Одновыборочный t-тест:")
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("✓ Отклоняем H0 - среднее значимо отличается от 100")
else:
    print("✗ Не можем отклонить H0")

# ДВУХВЫБОРОЧНЫЙ T-ТЕСТ (независимые выборки)
sample1 = np.random.normal(100, 15, 50)
sample2 = np.random.normal(102, 15, 50)

t_stat, p_value = ttest_ind(sample1, sample2)
print(f"\nДвухвыборочный t-тест:")
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("✓ Группы значимо отличаются")
else:
    print("✗ Различия не значимы")
```

### Проверка доли

```python
from scipy.stats import binom_test, binomtest

# H0: p = 0.5
# Ha: p ≠ 0.5

successes = 60
trials = 100
null_proportion = 0.5

# Биномиальный тест
result = binomtest(successes, trials, null_proportion, alternative='two-sided')
print(f"Биномиальный тест для доли:")
print(f"p-value = {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("✓ Доля значимо отличается от 0.5")
else:
    print("✗ Различия не значимы")
```

---

## ТЕМА 5: ПРОВЕРКА ГИПОТЕЗ (Непараметрические методы)

### Непараметрические тесты

```python
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, ranksums
import numpy as np

# MANN-WHITNEY U ТЕСТ (альтернатива t-тесту для независимых выборок)
# Проверяет: медианы двух групп отличаются?

sample1 = np.random.exponential(2, 50)  # Нормальное распределение нельзя предполагать
sample2 = np.random.exponential(2.5, 50)

u_stat, p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
print(f"Mann-Whitney U тест:")
print(f"U = {u_stat:.4f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("✓ Медианы группы значимо отличаются")

# WILCOXON SIGNED-RANK ТЕСТ (связные выборки)
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(2, 5, 30)  # Небольшое улучшение

w_stat, p_value = wilcoxon(before, after)
print(f"\nWilcoxon тест (связные выборки):")
print(f"W = {w_stat:.4f}, p-value = {p_value:.4f}")

# KRUSKAL-WALLIS ТЕСТ (несколько групп, непараметрический ANOVA)
group1 = np.random.exponential(2, 30)
group2 = np.random.exponential(2.3, 30)
group3 = np.random.exponential(2.5, 30)

h_stat, p_value = kruskal(group1, group2, group3)
print(f"\nKruskal-Wallis тест:")
print(f"H = {h_stat:.4f}, p-value = {p_value:.4f}")

# BOOTSTRAP (универсальный метод)
def bootstrap_mean_diff(sample1, sample2, n_bootstrap=10000):
    diffs = []
    for _ in range(n_bootstrap):
        sample1_boot = np.random.choice(sample1, len(sample1), replace=True)
        sample2_boot = np.random.choice(sample2, len(sample2), replace=True)
        diffs.append(np.mean(sample1_boot) - np.mean(sample2_boot))
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return ci_lower, ci_upper

ci_lower, ci_upper = bootstrap_mean_diff(sample1, sample2)
print(f"Bootstrap 95% CI для разности средних: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower < 0 < ci_upper:
    print("✗ CI содержит 0 → различия не значимы")
else:
    print("✓ CI не содержит 0 → различия значимы")
```

### Связные и несвязные выборки

```python
# НЕСВЯЗНЫЕ (независимые) выборки:
# - Разные люди в группах
# - Нет паирования
# Тесты: t-тест независимых, Mann-Whitney U

# СВЯЗНЫЕ (зависимые) выборки:
# - Один и тот же объект измерен дважды (до/после)
# - Попарно сопоставленные объекты
# Тесты: парный t-тест, Wilcoxon signed-rank

# Пример парного t-теста
from scipy.stats import ttest_rel

before_treatment = np.array([100, 102, 98, 101, 99])
after_treatment = np.array([98, 100, 95, 99, 97])

t_stat, p_value = ttest_rel(before_treatment, after_treatment)
print(f"Парный t-тест: t={t_stat:.4f}, p-value={p_value:.4f}")
```

### A/B тестирование

```python
# A/B тест - сравнение двух вариантов (A vs B)

# Вариант A: классический сайт
clicks_a = 45
impressions_a = 1000
ctr_a = clicks_a / impressions_a

# Вариант B: новый сайт
clicks_b = 60
impressions_b = 1000
ctr_b = clicks_b / impressions_b

# Двухвыборочный z-тест для пропорций
from statsmodels.stats.proportion import proportions_ztest

count = np.array([clicks_a, clicks_b])
nobs = np.array([impressions_a, impressions_b])

z_stat, p_value = proportions_ztest(count, nobs)
print(f"A/B тест:")
print(f"CTR A: {ctr_a:.4f}, CTR B: {ctr_b:.4f}")
print(f"z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Вариант B значимо лучше")
else:
    print("✗ Различия не значимы")
```

### Множественная проверка гипотез

```python
# Проблема: Если проверить 20 гипотез с α=0.05,
# вероятность ошибки первого рода растет!

# Решение 1: Bonferroni коррекция
n_tests = 20
alpha_bonferroni = 0.05 / n_tests  # 0.0025
print(f"Bonferroni alpha: {alpha_bonferroni:.4f}")

# Решение 2: FDR (False Discovery Rate)
from scipy.stats import norm

p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
m = len(p_values)
rank = np.arange(1, m + 1)
fdr_threshold = (rank / m) * 0.05

# Отклоняем гипотезы где p_value < fdr_threshold[rank-1]
print(f"FDR пороги: {fdr_threshold}")
```

---

## ТЕМА 6: РЕГРЕССИЯ

### Линейная регрессия

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Генерируем данные
X = np.random.randn(100, 1)
y = 2.5 * X.ravel() + np.random.randn(100) + 1

# SKLEARN
model = LinearRegression()
model.fit(X, y)

print(f"Коэффициент (наклон): {model.coef_[0]:.4f}")
print(f"Intercept (пересечение): {model.intercept_:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")

# Предсказания
y_pred = model.predict(X)

# Остатки
residuals = y - y_pred

# MAE, MSE, RMSE
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
```

### Регрессия с TensorFlow

```python
import tensorflow as tf
import numpy as np

# Генерируем данные
X = np.random.randn(100, 1).astype(np.float32)
y = 2.5 * X.ravel() + np.random.randn(100).astype(np.float32) + 1

# Модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Linear output
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)

# Предсказания
y_pred = model.predict(X).flatten()
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print(f"R² TensorFlow: {r2:.4f}")
```

### Авторегрессия (AR)

```python
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import pandas as pd

# Временной ряд
np.random.seed(42)
data = np.cumsum(np.random.randn(100))

# AutoReg модель
model = AutoReg(data, lags=5)  # использует 5 предыдущих значений
fitted_model = model.fit()

print(fitted_model.summary())

# Предсказания
predictions = fitted_model.predict(start=5, end=99)
```

---

## ТЕМА 7: МЕТОДЫ ОЦЕНКИ ЗНАЧИМОСТИ ПАРАМЕТРОВ

```python
from scipy import stats
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# МЕТОД 1: T-СТАТИСТИКА И P-VALUE

X = np.random.randn(100, 1)
y = 2.5 * X.ravel() + np.random.randn(100) + 1

# Добавляем intercept
X_with_const = np.column_stack([np.ones(len(X)), X])

# Коэффициенты через МНК (Least Squares)
beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y

# Остатки и стандартная ошибка
residuals = y - X_with_const @ beta
s_squared = np.sum(residuals**2) / (len(y) - X_with_const.shape[1])

# Матрица ковариации
var_beta = s_squared * np.linalg.inv(X_with_const.T @ X_with_const)
se_beta = np.sqrt(np.diag(var_beta))

# T-статистика
t_stats = beta / se_beta

# P-values (двусторонний тест)
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - 2))

print(f"Коэффициенты: {beta}")
print(f"Стандартные ошибки: {se_beta}")
print(f"T-статистика: {t_stats}")
print(f"P-values: {p_values}")

# МЕТОД 2: Статистика с использованием statsmodels
import statsmodels.api as sm

X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
results = model.fit()
print(results.summary())

# МЕТОД 3: Bootstrap для доверительных интервалов
def bootstrap_ci(X, y, n_bootstrap=1000):
    n = len(X)
    coefficients = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        X_boot_const = np.column_stack([np.ones(n), X_boot])
        beta_boot = np.linalg.inv(X_boot_const.T @ X_boot_const) @ X_boot_const.T @ y_boot
        coefficients.append(beta_boot[1])  # только slope
    
    coefficients = np.array(coefficients)
    ci_lower = np.percentile(coefficients, 2.5)
    ci_upper = np.percentile(coefficients, 97.5)
    
    return ci_lower, ci_upper

ci_l, ci_u = bootstrap_ci(X, y)
print(f"\nBootstrap 95% CI для коэффициента: [{ci_l:.4f}, {ci_u:.4f}]")
```

---

## ТЕМА 8: КЛАСТЕРИЗАЦИЯ

### KMeans

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

# Генерируем данные
X = np.random.randn(300, 2)
X[:100] += np.array([5, 5])
X[100:200] += np.array([10, 0])

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Центры: {kmeans.cluster_centers_}")
print(f"Инерция (сумма квадратов внутри): {kmeans.inertia_:.4f}")
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Нормализуем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Кластеров: {n_clusters}")
print(f"Шумовых точек: {n_noise}")
```

### Иерархическая кластеризация

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Связывание методом Уорда (Ward)
Z = linkage(X, method='ward')

# Дендрограмма
dendrogram(Z)

# Получить кластеры с дистанцией 5
clusters = fcluster(Z, t=5, criterion='distance')
print(f"Кластеры: {np.unique(clusters)}")
```

### Оценка качества кластеризации

```python
# ВНУТРЕННИЕ МЕТРИКИ (без истинных меток)

# Silhouette Score (-1 до 1, выше лучше)
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Davies-Bouldin Index (ниже лучше)
db_index = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_index:.4f}")

# Calinski-Harabasz Index (выше лучше)
from sklearn.metrics import calinski_harabasz_score
ch_index = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {ch_index:.4f}")

# ВНЕШНИЕ МЕТРИКИ (с истинными метками)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

true_labels = np.array([0]*100 + [1]*100 + [2]*100)

ari = adjusted_rand_score(true_labels, labels)
nmi = normalized_mutual_info_score(true_labels, labels)

print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Info: {nmi:.4f}")
```

### Выбор числа кластеров

```python
# Метод локтя (Elbow Method)
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

# Оптимальный k - где начинается "локоть"
plt.plot(K_range, inertias, 'o-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
```

---

## ТЕМА 9: ТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ (Topic Modeling)

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Документы
documents = [
    "Python machine learning deep learning",
    "data science statistics analysis",
    "neural networks AI artificial intelligence",
    "clustering classification supervised learning",
    "regression prediction model"
]

# Создаем матрицу терминов-документов
vectorizer = CountVectorizer(max_features=20, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

# LDA модель (Latent Dirichlet Allocation)
lda = LatentDirichletAllocation(
    n_components=2,  # 2 темы
    random_state=42,
    max_iter=20
)

lda.fit(doc_term_matrix)

# Основные слова в каждой теме
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Тема {topic_idx}: {', '.join(top_words)}")

# Распределение тем по документам
doc_topic_dist = lda.transform(doc_term_matrix)
print(f"Распределение тем в первом документе: {doc_topic_dist[0]}")
```

---

## ТЕМА 10: ОРТОГОНАЛЬНЫЕ МАТРИЧНЫЕ ПРЕОБРАЗОВАНИЯ

### Преобразование Хаара

```python
import numpy as np
from scipy.fftpack import dct

# Преобразование Хаара (простое ортогональное преобразование)
def haar_transform(signal):
    """Одномерное преобразование Хаара"""
    n = len(signal)
    if n == 1:
        return signal
    
    # Divide and average
    averages = (signal[::2] + signal[1::2]) / np.sqrt(2)
    differences = (signal[::2] - signal[1::2]) / np.sqrt(2)
    
    return np.concatenate([averages, differences])

def inverse_haar_transform(transformed):
    """Обратное преобразование Хаара"""
    n = len(transformed)
    half = n // 2
    
    averages = transformed[:half]
    differences = transformed[half:]
    
    signal = np.zeros(n)
    signal[::2] = (averages + differences) / np.sqrt(2)
    signal[1::2] = (averages - differences) / np.sqrt(2)
    
    return signal

# Пример
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
transformed = haar_transform(signal)
reconstructed = inverse_haar_transform(transformed)

print(f"Исходный сигнал: {signal}")
print(f"После Хаара: {transformed}")
print(f"Восстановленный: {reconstructed}")
```

### Преобразование Уолша

```python
# Матрица Адамара (основа Уолша)
def hadamard_matrix(n):
    """Создает матрицу Адамара размером n x n (n = 2^k)"""
    if n == 1:
        return np.array([[1]])
    
    H = hadamard_matrix(n // 2)
    return np.vstack([
        np.hstack([H, H]),
        np.hstack([H, -H])
    ]) / np.sqrt(2)

# Преобразование Уолша
def walsh_transform(signal):
    n = len(signal)
    W = hadamard_matrix(n)
    return W @ signal

# Спектр Уолша (коэффициенты)
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
walsh_coeff = walsh_transform(signal)

print(f"Коэффициенты Уолша: {walsh_coeff}")
```

---

## ТЕМА 11: ПРЕОБРАЗОВАНИЕ ФУРЬЕ

```python
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

# Сигнал: сумма 3 синусоид
t = np.linspace(0, 1, 256, endpoint=False)
signal = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*15*t)

# Прямое преобразование Фурье
fft_coeffs = fft(signal)
freqs = fftfreq(len(signal), t[1]-t[0])

# Спектр мощности (амплитуды)
power = np.abs(fft_coeffs) ** 2

# Только положительные частоты
positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]

# Основные частоты
top_freqs = positive_freqs[np.argsort(positive_power)[-3:]]
print(f"Основные частоты: {np.sort(top_freqs)}")

# Обратное преобразование
reconstructed = np.real(ifft(fft_coeffs))
print(f"Ошибка восстановления: {np.max(np.abs(signal - reconstructed)):.10f}")

# Спектрограмма (Fourier Spectrogram)
from scipy.signal import spectrogram
f, t_spec, Sxx = spectrogram(signal, nperseg=64)
print(f"Частоты спектрограммы: {f}")
print(f"Время спектрограммы: {t_spec}")
```

---

## ТЕМА 12: ОКОННОЕ ПРЕОБРАЗОВАНИЕ ФУРЬЕ И ВРЕМЕННО-ЧАСТОТНЫЙ АНАЛИЗ

```python
from scipy.signal import stft, istft
from scipy.signal.windows import hann
import numpy as np
import matplotlib.pyplot as plt

# Сигнал с изменяющимися частотами
t = np.linspace(0, 2, 512)
signal = np.sin(2*np.pi*5*t) * (t < 1) + np.sin(2*np.pi*20*t) * (t >= 1)

# Оконное преобразование Фурье (Short-Time Fourier Transform)
f, t_stft, Zxx = stft(signal, fs=256, window='hann', nperseg=64)

print(f"STFT shape: {Zxx.shape}")  # (frequency, time)

# Реконструкция сигнала
t_recon, signal_recon = istft(Zxx, fs=256, window='hann', nperseg=64)
print(f"Ошибка: {np.max(np.abs(signal - signal_recon[:len(signal)])):.10f}")

# ARIMA модель для временных рядов
from statsmodels.tsa.arima.model import ARIMA

# Генерируем временной ряд
np.random.seed(42)
ts = np.cumsum(np.random.randn(100))

# ARIMA(1,1,1) - AutoRegressive Integrated Moving Average
model = ARIMA(ts, order=(1, 1, 1))
fitted_model = model.fit()

print(fitted_model.summary())

# Прогноз
forecast = fitted_model.get_forecast(steps=10)
print(f"Прогноз: {forecast.predicted_mean.values}")

# SARIMAX (с сезонностью)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMAX(1,1,1)x(1,1,1,12) - с сезонным компонентом (период=12)
model_seasonal = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
fitted_seasonal = model_seasonal.fit()
```

---

## ТЕМА 13: ВЫБРОСЫ И АВТОЭНКОДЕРЫ

### Методы обнаружения выбросов

```python
import numpy as np
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

# Данные с выбросами
X = np.random.randn(100, 2)
X = np.vstack([X, np.array([[10, 10], [10, -10], [-10, 10]])])

# МЕТОД 1: Z-Score
z_scores = np.abs(stats.zscore(X))
outliers_zscore = (z_scores > 3).any(axis=1)
print(f"Z-Score выбросы: {np.sum(outliers_zscore)}")

# МЕТОД 2: IQR (Interquartile Range)
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
outliers_iqr = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"IQR выбросы: {np.sum(outliers_iqr)}")

# МЕТОД 3: Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers_if = iso_forest.fit_predict(X) == -1
print(f"Isolation Forest выбросы: {np.sum(outliers_if)}")

# МЕТОД 4: Elliptic Envelope (Mahalanobis distance)
elliptic = EllipticEnvelope(contamination=0.05, random_state=42)
outliers_elliptic = elliptic.fit_predict(X) == -1
print(f"Elliptic Envelope выбросы: {np.sum(outliers_elliptic)}")
```

### Автоэнкодеры

```python
import tensorflow as tf
import numpy as np

# Генерируем данные
normal_data = np.random.randn(1000, 10)
anomaly_data = np.random.uniform(-5, 5, (50, 10))  # Выбросы
X_train = normal_data[:900]
X_val_normal = normal_data[900:]
X_val_anomaly = anomaly_data

# Автоэнкодер
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(4, activation='relu'),  # Bottleneck
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, verbose=0)

# Вычисляем ошибку восстановления (reconstruction error)
train_predictions = autoencoder.predict(X_train)
train_mse = np.mean(np.square(X_train - train_predictions), axis=1)
threshold = np.percentile(train_mse, 95)  # 95-й перцентиль

# Тестируем на нормальных и аномальных данных
val_normal_pred = autoencoder.predict(X_val_normal)
val_anomaly_pred = autoencoder.predict(X_val_anomaly)

mse_normal = np.mean(np.square(X_val_normal - val_normal_pred), axis=1)
mse_anomaly = np.mean(np.square(X_val_anomaly - val_anomaly_pred), axis=1)

print(f"Порог: {threshold:.4f}")
print(f"Средняя ошибка нормальных: {np.mean(mse_normal):.4f}")
print(f"Средняя ошибка аномалий: {np.mean(mse_anomaly):.4f}")

# Классификация
anomaly_detected_normal = np.sum(mse_normal > threshold)
anomaly_detected_anomaly = np.sum(mse_anomaly > threshold)
print(f"Ложные срабатывания: {anomaly_detected_normal}/{len(mse_normal)}")
print(f"Верные обнаружения: {anomaly_detected_anomaly}/{len(mse_anomaly)}")
```

---

## ТЕМА 14: ИЗВЛЕЧЕНИЕ ОСОБЕННОСТЕЙ (Feature Extraction)

### Понижение размерности (Dimensionality Reduction)

```python
import numpy as np
from sklearn.decomposition import PCA, TSNE
from sklearn.manifold import UMAP

# Высокомерные данные
X = np.random.randn(1000, 100)  # 1000 объектов, 100 признаков

# МЕТОД 1: PCA (линейный метод)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Объясненная вариация: {pca.explained_variance_ratio_}")
print(f"Сумма: {np.sum(pca.explained_variance_ratio_):.4f}")

# Выбор количества компонент
pca_full = PCA()
pca_full.fit(X)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1  # 95% вариации
print(f"Компонент для 95% вариации: {n_components}")

# МЕТОД 2: t-SNE (нелинейный метод)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# МЕТОД 3: UMAP (быстрее t-SNE)
umap_reducer = UMAP(n_components=2)
X_umap = umap_reducer.fit_transform(X)
```

### Повышение размерности (Feature Generation)

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Исходные признаки
X = np.random.randn(100, 2)

# Полиномиальные признаки
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"Исходная форма: {X.shape}")
print(f"После полиномиальных признаков: {X_poly.shape}")
# Содержит: [X1, X2, X1^2, X1*X2, X2^2]

# Другие трансформации
# Log, sqrt, exp трансформации
X_log = np.log1p(np.abs(X))
X_sqrt = np.sqrt(np.abs(X))

# Взаимодействия
from sklearn.preprocessing import PolynomialFeatures
interaction_terms = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interaction = interaction_terms.fit_transform(X)
```

---

## ТЕМА 15: ОТТОК (Churn Prediction)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

# Симулируем данные об отчислении студентов
np.random.seed(42)
n_students = 1000

data = pd.DataFrame({
    'gpa': np.random.normal(3.2, 0.6, n_students),
    'attendance': np.random.uniform(0.5, 1, n_students),
    'assignment_completion': np.random.uniform(0, 1, n_students),
    'library_visits': np.random.poisson(10, n_students),
    'office_hours': np.random.poisson(5, n_students),
    'family_income': np.random.exponential(50000, n_students)
})

# Целевая переменная: отток (1 = отчислился/отказался, 0 = продолжает)
# Чем выше GPA и посещаемость, тем ниже риск отчисления
churn_prob = 0.9 - (data['gpa'] / 5) * 0.3 - (data['attendance'] * 0.2)
churn_prob = np.clip(churn_prob, 0, 1)
data['churn'] = np.random.binomial(1, churn_prob)

print(f"Уровень оттока: {data['churn'].mean():.2%}")

# TRAIN-TEST SPLIT
X = data.drop('churn', axis=1)
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализуем
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# МОДЕЛЬ: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ПРЕДСКАЗАНИЯ
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

# ОЦЕНКИ
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Кривая Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nВажность признаков:\n{feature_importance}")

# ИНТЕРПРЕТАЦИЯ
print("\nСтратегия удержания студентов:")
print("1. Мониторить GPA - главный фактор оттока")
print("2. Улучшить посещаемость на лекциях")
print("3. Отслеживать студентов с низким GPA (<2.5)")
print("4. Предложить дополнительные консультации на базе модели")
```

---

## ⚡ БЫСТРАЯ СПРАВКА НА ЭКЗАМЕНЕ

### Что выводить на экзамене

```
КОД:
1. Что делает каждая функция?
2. Входные параметры
3. Выходные данные
4. Физический смысл

ОШИБКА:
- Прочитать строки 100-150
- Найти синтаксис ошибку ИЛИ логическую ошибку
- Предложить исправление

МЕТОД:
- Какой статистический тест?
- Какой алгоритм?
- Какие исключения?

ВЫВОД:
- Нужна ли нормализация?
- Есть ли выбросы?
- Что будет на печати?
```

### Часто встречаются темы

```python
# Всегда нужны:
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing, model_selection, metrics
import matplotlib.pyplot as plt

# Основные тесты
stats.ttest_ind(a, b)  # Сравнение 2 групп
stats.f_oneway(*groups)  # Сравнение 3+ групп
stats.chi2_contingency(table)  # Категориальные признаки
stats.pearsonr(x, y)  # Корреляция

# Машинное обучение
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
```

### Типичные вопросы

1. **"Какой метод?"** → Смотри входные данные (числовые/категориальные)
2. **"Ошибка?"** → Синтаксис + логика (индексы, типы, размеры)
3. **"Вывод?"** → Трассируй переменные пошагово
4. **"Конспект?"** → Назови: Что? Зачем? Как? Когда?
