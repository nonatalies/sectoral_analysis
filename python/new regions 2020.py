"""Статистика регионы отдельно 2020 год"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tabulate import tabulate

def load_data():
    new_regions = pd.read_excel('Данные для стат анализа регионы отдельно.xlsx',
                              sheet_name='Лист1', header=2)
    all_regions = pd.read_excel('Данные для стат анализа РФ отдельно.xlsx',
                              sheet_name='Лист1', header=2)

    new_regions.columns = [
        'Регион',
        'Посевные площади (тыс. га)',
        'Индекс сельского хозяйства',
        'Добыча полезных ископаемых (млн руб)',
        'Индекс пром. производства',
        'Потребление электроэнергии (млн кВт·ч)',
        'Производство электроэнергии на душу (кВт·ч/чел)'
    ]

    new_regions = new_regions.dropna(subset=['Регион'])
    all_regions = all_regions.dropna(subset=['НАИМЕНОВАНИЕ РЕГИОНА'])

    all_regions.columns = [str(col) for col in all_regions.columns]

    return new_regions, all_regions

def calculate_metrics(new_regions):
    metrics = {}

    numeric_cols = new_regions.select_dtypes(include=[np.number]).columns

    # 1. Базовые статистики по каждому показателю
    for col in numeric_cols:
        metrics[f'Среднее {col}'] = new_regions[col].mean()
        metrics[f'Медиана {col}'] = new_regions[col].median()
        metrics[f'Стандартное отклонение {col}'] = new_regions[col].std()
        metrics[f'Коэффициент вариации {col} (%)'] = (new_regions[col].std() / new_regions[col].mean()) * 100
        metrics[f'Минимальное значение {col}'] = new_regions[col].min()
        metrics[f'Максимальное значение {col}'] = new_regions[col].max()
        metrics[f'Квартиль 25% {col}'] = new_regions[col].quantile(0.25)
        metrics[f'Квартиль 75% {col}'] = new_regions[col].quantile(0.75)

    # 2. Показатели концентрации и неравномерности
    metrics['Коэффициент Джини (добыча)'] = gini_coefficient(new_regions['Добыча полезных ископаемых (млн руб)'])
    metrics['Коэффициент Джини (посевные площади)'] = gini_coefficient(new_regions['Посевные площади (тыс. га)'])
    metrics['Индекс Херфиндаля-Хиршмана (добыча)'] = herfindahl_index(new_regions['Добыча полезных ископаемых (млн руб)'])

    # 3. Экономическая эффективность
    metrics['Энергоемкость добычи (кВт·ч/млн руб)'] = (
        new_regions['Потребление электроэнергии (млн кВт·ч)'].sum() /
        new_regions['Добыча полезных ископаемых (млн руб)'].sum()
    )

    metrics['Производительность сельского хозяйства (усл.ед./га)'] = (
        new_regions['Индекс сельского хозяйства'] /
        new_regions['Посевные площади (тыс. га)']
    ).mean()

    # 4. Показатели диверсификации
    metrics['Коэффициент диверсификации экономики'] = calculate_diversification(new_regions)

    # 5. Кластерный анализ
    cluster_labels = cluster_regions(new_regions)
    new_regions['Кластер'] = cluster_labels
    metrics['Распределение по кластерам'] = dict(new_regions['Кластер'].value_counts())

    # 6. Корреляции между ключевыми показателями
    corr_matrix = new_regions[numeric_cols].corr()
    metrics['Корреляция добыча-энергопотребление'] = corr_matrix.loc[
        'Добыча полезных ископаемых (млн руб)', 'Потребление электроэнергии (млн кВт·ч)']
    metrics['Корреляция посевы-сельхоз индекс'] = corr_matrix.loc[
        'Посевные площади (тыс. га)', 'Индекс сельского хозяйства']

    return metrics, new_regions

def gini_coefficient(x):
    """Расчет коэффициента Джини для оценки неравномерности распределения"""
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def herfindahl_index(x):
    """Индекс Херфиндаля-Хиршмана для оценки концентрации"""
    x = np.array(x)
    x = x[x > 0]  
    s = x.sum()
    return np.sum((x / s) ** 2) * 10000

def calculate_diversification(df):
    """Коэффициент диверсификации экономики"""
    indicators = [
        'Добыча полезных ископаемых (млн руб)',
        'Посевные площади (тыс. га)',
        'Потребление электроэнергии (млн кВт·ч)'
    ]
    data = df[indicators].copy()
    data = data / data.sum() 

    entropy = -np.sum(data * np.log(data + 1e-10), axis=1).mean()
    return entropy

def cluster_regions(df, n_clusters=3):
    """Кластеризация регионов по экономическим показателям"""
    features = df.select_dtypes(include=[np.number]).columns
    X = df[features].fillna(0)
    X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X)

def print_data_table(title, data, headers):
    print(f"\n=== {title} ===")
    print(tabulate(data, headers=headers, tablefmt='grid', floatfmt=".2f"))

def visualize_comparison(new_regions):
    # 1. Посевные площади
    plt.figure(figsize=(12, 6))
    plot_data = new_regions[['Регион', 'Посевные площади (тыс. га)']].sort_values('Посевные площади (тыс. га)')
    sns.barplot(x='Регион', y='Посевные площади (тыс. га)', data=plot_data)
    plt.title('Посевные площади в новых регионах')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print_data_table(
        "Посевные площади по регионам",
        plot_data.values,
        ["Регион", "Посевные площади (тыс. га)"]
    )

    # 2. Индексы промышленного производства
    plt.figure(figsize=(12, 6))
    plot_data = new_regions[['Регион', 'Индекс пром. производства']].sort_values('Индекс пром. производства')
    sns.barplot(x='Регион', y='Индекс пром. производства', data=plot_data)
    plt.axhline(y=100, color='red', linestyle='--', label='Базовый уровень (100%)')
    plt.title('Индекс промышленного производства в новых регионах')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print_data_table(
        "Индекс промышленного производства",
        plot_data.values,
        ["Регион", "Индекс пром. производства"]
    )

    # 3. Энергетические показатели
    plt.figure(figsize=(10, 6))
    plot_data = new_regions[['Регион', 'Потребление электроэнергии (млн кВт·ч)',
                           'Производство электроэнергии на душу (кВт·ч/чел)']]
    sns.scatterplot(
        x='Потребление электроэнергии (млн кВт·ч)',
        y='Производство электроэнергии на душу (кВт·ч/чел)',
        data=plot_data,
        hue='Регион',
        s=200
    )
    plt.title('Соотношение потребления и производства электроэнергии')
    plt.tight_layout()
    plt.show()

    print_data_table(
        "Энергетические показатели",
        plot_data.values,
        ["Регион", "Потребление (млн кВт·ч)", "Производство на душу (кВт·ч/чел)"]
    )

    # 4. Распределение показателей с ящиками с усами
    plt.figure(figsize=(12, 6))
    numeric_cols = new_regions.select_dtypes(include=[np.number]).columns
    melted = new_regions.melt(id_vars=['Регион'], value_vars=numeric_cols)
    sns.boxplot(x='variable', y='value', data=melted)
    plt.xticks(rotation=45)
    plt.title('Распределение показателей по новым регионам')
    plt.tight_layout()
    plt.show()

    stats_table = new_regions[numeric_cols].describe().T.round(2)
    print("\n=== Описательные статистики по показателям ===")
    print(tabulate(stats_table, headers=[''] + list(stats_table.columns), tablefmt='grid'))

    # 5. Карта корреляций
    corr = new_regions[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Корреляция показателей в новых регионах')
    plt.tight_layout()
    plt.show()

    print("\n=== Матрица корреляции ===")
    print(tabulate(corr.round(2), headers=[''] + list(corr.columns), tablefmt='grid'))

    # 6. Кластерный анализ 
    if 'Кластер' in new_regions.columns:
        plt.figure(figsize=(10, 8))
        plot_data = new_regions[['Регион', 'Добыча полезных ископаемых (млн руб)',
                               'Потребление электроэнергии (млн кВт·ч)', 'Кластер']]
        sns.scatterplot(
            x='Добыча полезных ископаемых (млн руб)',
            y='Потребление электроэнергии (млн кВт·ч)',
            hue='Кластер',
            data=plot_data,
            palette='viridis',
            s=100
        )
        plt.title('Кластеризация регионов по добыче и энергопотреблению')
        plt.tight_layout()
        plt.show()

        print_data_table(
            "Кластерный анализ",
            plot_data.values,
            ["Регион", "Добыча (млн руб)", "Потребление (млн кВт·ч)", "Кластер"]
        )

def main_analysis():
    new_regions, _ = load_data()

    metrics, new_regions = calculate_metrics(new_regions)

    print("\n=== Ключевые метрики новых регионов ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    visualize_comparison(new_regions)

if __name__ == "__main__":
    main_analysis()
