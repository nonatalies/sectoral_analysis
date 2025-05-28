"""Статистика РФ отдельно 2020 год по округам"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tabulate import tabulate

def load_rf_data():
    rf_data = pd.read_excel('Данные для стат анализа РФ отдельно.xlsx', sheet_name='Лист1', header=2)
    rf_data = rf_data.dropna(subset=['НАИМЕНОВАНИЕ РЕГИОНА'])
    rf_data.columns = [str(col) for col in rf_data.columns]
    return rf_data

def prepare_rf_data(rf_data):
    federal_districts = [
        'Центральный федеральный округ', 'Северо-Западный федеральный округ',
        'Южный федеральный округ', 'Северо-Кавказский федеральный округ',
        'Приволжский федеральный округ', 'Уральский федеральный округ',
        'Сибирский федеральный округ', 'Дальневосточный федеральный округ'
    ]

    analysis_data = rf_data[rf_data['НАИМЕНОВАНИЕ РЕГИОНА'].isin(federal_districts)].copy()

    analysis_data = analysis_data.rename(columns={
        'НАИМЕНОВАНИЕ РЕГИОНА': 'Регион',
        '2020': 'Посевные площади (тыс. га)',
        '2020.1': 'Индекс сельского хозяйства',
        '2020.2': 'Добыча полезных ископаемых (млн руб)',
        '2020.3': 'Индекс пром. производства',
        '2020.4': 'Потребление электроэнергии (млн кВт·ч)',
        '2020.5': 'Производство электроэнергии на душу (кВт·ч/чел)'
    })

    new_regions = analysis_data[['Регион',
                               'Посевные площади (тыс. га)',
                               'Индекс сельского хозяйства',
                               'Добыча полезных ископаемых (млн руб)',
                               'Индекс пром. производства',
                               'Потребление электроэнергии (млн кВт·ч)',
                               'Производство электроэнергии на душу (кВт·ч/чел)']].copy()

    return new_regions

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

    missing_cols = [col for col in indicators if col not in df.columns]
    if missing_cols:
        print(f"Предупреждение: отсутствуют столбцы {missing_cols} для расчета диверсификации")
        return np.nan

    try:
        data = df[indicators].replace(0, 1e-10).copy()

        row_sums = data.sum(axis=1)
        normalized = data.div(row_sums, axis=0)

        entropy = -np.sum(normalized * np.log(normalized), axis=1)

        return entropy.mean()
    except Exception as e:
        print(f"Ошибка при расчете диверсификации: {e}")
        return np.nan

def cluster_regions(df, n_clusters=3):
    """Кластеризация регионов по экономическим показателям"""
    features = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Кластер']

    if not features:
        print("Нет числовых признаков для кластеризации")
        return np.zeros(len(df))

    try:
        X = df[features].fillna(0)
        if len(X) < n_clusters:
            print(f"Слишком мало регионов ({len(X)}) для кластеризации на {n_clusters} кластера")
            return np.zeros(len(X))

        X = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(X)
    except Exception as e:
        print(f"Ошибка при кластеризации: {e}")
        return np.zeros(len(df))

def calculate_metrics(new_regions):
    metrics = {}

    numeric_cols = [col for col in new_regions.select_dtypes(include=[np.number]).columns if col != 'Кластер']

    # 1. Базовые статистики
    for col in numeric_cols:
        metrics[f'Среднее {col}'] = new_regions[col].mean()
        metrics[f'Медиана {col}'] = new_regions[col].median()
        metrics[f'Стандартное отклонение {col}'] = new_regions[col].std()
        cv = (new_regions[col].std() / new_regions[col].mean()) * 100 if new_regions[col].mean() != 0 else np.nan
        metrics[f'Коэффициент вариации {col} (%)'] = cv
        metrics[f'Минимальное значение {col}'] = new_regions[col].min()
        metrics[f'Максимальное значение {col}'] = new_regions[col].max()
        metrics[f'Квартиль 25% {col}'] = new_regions[col].quantile(0.25)
        metrics[f'Квартиль 75% {col}'] = new_regions[col].quantile(0.75)

    # 2. Показатели концентрации
    if 'Добыча полезных ископаемых (млн руб)' in new_regions.columns:
        metrics['Коэффициент Джини (добыча)'] = gini_coefficient(new_regions['Добыча полезных ископаемых (млн руб)'])
    if 'Посевные площади (тыс. га)' in new_regions.columns:
        metrics['Коэффициент Джини (посевные площади)'] = gini_coefficient(new_regions['Посевные площади (тыс. га)'])
    if 'Добыча полезных ископаемых (млн руб)' in new_regions.columns:
        metrics['Индекс Херфиндаля-Хиршмана (добыча)'] = herfindahl_index(new_regions['Добыча полезных ископаемых (млн руб)'])

    # 3. Экономическая эффективность
    if 'Потребление электроэнергии (млн кВт·ч)' in new_regions.columns and 'Добыча полезных ископаемых (млн руб)' in new_regions.columns:
        total_energy = new_regions['Потребление электроэнергии (млн кВт·ч)'].sum()
        total_mining = new_regions['Добыча полезных ископаемых (млн руб)'].sum()
        metrics['Энергоемкость добычи (кВт·ч/млн руб)'] = total_energy / total_mining if total_mining != 0 else np.nan

    if 'Индекс сельского хозяйства' in new_regions.columns and 'Посевные площади (тыс. га)' in new_regions.columns:
        agri_index = new_regions['Индекс сельского хозяйства']
        crop_area = new_regions['Посевные площади (тыс. га)']
        metrics['Производительность сельского хозяйства (усл.ед./га)'] = (agri_index / crop_area.replace(0, np.nan)).mean()

    # 4. Показатели диверсификации
    metrics['Коэффициент диверсификации экономики'] = calculate_diversification(new_regions)

    # 5. Кластерный анализ
    cluster_labels = cluster_regions(new_regions)
    if cluster_labels is not None:
        new_regions['Кластер'] = cluster_labels
        metrics['Распределение по кластерам'] = dict(new_regions['Кластер'].value_counts())

    # 6. Корреляции
    if len(numeric_cols) > 1:
        corr_matrix = new_regions[numeric_cols].corr()
        if 'Добыча полезных ископаемых (млн руб)' in numeric_cols and 'Потребление электроэнергии (млн кВт·ч)' in numeric_cols:
            metrics['Корреляция добыча-энергопотребление'] = corr_matrix.loc[
                'Добыча полезных ископаемых (млн руб)', 'Потребление электроэнергии (млн кВт·ч)']
        if 'Посевные площади (тыс. га)' in numeric_cols and 'Индекс сельского хозяйства' in numeric_cols:
            metrics['Корреляция посевы-сельхоз индекс'] = corr_matrix.loc[
                'Посевные площади (тыс. га)', 'Индекс сельского хозяйства']

    return metrics, new_regions

def print_data_table(title, data, headers):
    print(f"\n=== {title} ===")
    print(tabulate(data, headers=headers, tablefmt='grid', floatfmt=".2f"))

def visualize_comparison(new_regions):
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # 1. Посевные площади
    if 'Посевные площади (тыс. га)' in new_regions.columns:
        plt.figure(figsize=(12, 6))
        plot_data = new_regions[['Регион', 'Посевные площади (тыс. га)']].sort_values('Посевные площади (тыс. га)')
        sns.barplot(x='Регион', y='Посевные площади (тыс. га)', data=plot_data)
        plt.title('Посевные площади по федеральным округам РФ (2020)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print_data_table(
            "Посевные площади по регионам (2020)",
            plot_data.values,
            ["Регион", "Посевные площади (тыс. га)"]
        )

    # 2. Индексы промышленного производства
    if 'Индекс пром. производства' in new_regions.columns:
        plt.figure(figsize=(12, 6))
        plot_data = new_regions[['Регион', 'Индекс пром. производства']].sort_values('Индекс пром. производства')
        sns.barplot(x='Регион', y='Индекс пром. производства', data=plot_data)
        plt.axhline(y=100, color='red', linestyle='--', label='Базовый уровень (100%)')
        plt.title('Индекс промышленного производства по федеральным округам РФ (2020)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print_data_table(
            "Индекс промышленного производства (2020)",
            plot_data.values,
            ["Регион", "Индекс пром. производства"]
        )

    # 3. Энергетические показатели
    if 'Потребление электроэнергии (млн кВт·ч)' in new_regions.columns and 'Производство электроэнергии на душу (кВт·ч/чел)' in new_regions.columns:
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
        plt.title('Соотношение потребления и производства электроэнергии (2020)')
        plt.tight_layout()
        plt.show()

        print_data_table(
            "Энергетические показатели (2020)",
            plot_data.values,
            ["Регион", "Потребление (млн кВт·ч)", "Производство на душу (кВт·ч/чел)"]
        )

    # 4. Распределение показателей
    numeric_cols = [col for col in new_regions.select_dtypes(include=[np.number]).columns if col != 'Кластер']
    if numeric_cols:
        plt.figure(figsize=(12, 6))
        melted = new_regions.melt(id_vars=['Регион'], value_vars=numeric_cols)
        sns.boxplot(x='variable', y='value', data=melted)
        plt.xticks(rotation=45)
        plt.title('Распределение показателей по федеральным округам РФ (2020)')
        plt.tight_layout()
        plt.show()

        stats_table = new_regions[numeric_cols].describe().T.round(2)
        print("\n=== Описательные статистики по показателям (2020) ===")
        print(tabulate(stats_table, headers=[''] + list(stats_table.columns), tablefmt='grid'))

    # 5. Карта корреляций
    if len(numeric_cols) > 1:
        corr = new_regions[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
        plt.title('Корреляция показателей по федеральным округам РФ (2020)')
        plt.tight_layout()
        plt.show()

        print("\n=== Матрица корреляции (2020) ===")
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
        plt.title('Кластеризация федеральных округов по добыче и энергопотреблению (2020)')
        plt.tight_layout()
        plt.show()

        print_data_table(
            "Кластерный анализ (2020)",
            plot_data.values,
            ["Регион", "Добыча (млн руб)", "Потребление (млн кВт·ч)", "Кластер"]
        )

def main_rf_analysis():
    """Основная функция анализа"""
    rf_data = load_rf_data()
    new_regions = prepare_rf_data(rf_data)

    metrics, new_regions = calculate_metrics(new_regions)

    print("\n=== Ключевые метрики федеральных округов РФ за 2020 год ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        elif isinstance(value, dict):
            print(f"\n{name}:")
            for k, v in value.items():
                print(f"  Кластер {k}: {v} регионов")
        else:
            print(f"{name}: {value}")

    visualize_comparison(new_regions)

if __name__ == "__main__":
    main_rf_analysis()
