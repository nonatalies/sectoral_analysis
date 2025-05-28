"""Сравнение существующих и новых"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tabulate import tabulate

plt.style.use('seaborn-v0_8')  
sns.set_theme(style="whitegrid") 
sns.set_palette("husl")

def load_data():
    """Загрузка данных для новых и существующих регионов"""
    try:

        new_regions = pd.read_excel('Данные для стат анализа регионы отдельно.xlsx',
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
        new_regions['Тип региона'] = 'Новый'

        rf_data = pd.read_excel('Данные для стат анализа РФ отдельно без деления на округа.xlsx',
                               sheet_name='Лист1', header=2)
        rf_data = rf_data.dropna(subset=['НАИМЕНОВАНИЕ РЕГИОНА'])
        rf_data.columns = [str(col) for col in rf_data.columns]

        rf_data = rf_data.rename(columns={
            'НАИМЕНОВАНИЕ РЕГИОНА': 'Регион',
            '2020': 'Посевные площади (тыс. га)',
            '2020.1': 'Индекс сельского хозяйства',
            '2020.2': 'Добыча полезных ископаемых (млн руб)',
            '2020.3': 'Индекс пром. производства',
            '2020.4': 'Потребление электроэнергии (млн кВт·ч)',
            '2020.5': 'Производство электроэнергии на душу (кВт·ч/чел)'
        })

        existing_regions = rf_data[['Регион',
                                 'Посевные площади (тыс. га)',
                                 'Индекс сельского хозяйства',
                                 'Добыча полезных ископаемых (млн руб)',
                                 'Индекс пром. производства',
                                 'Потребление электроэнергии (млн кВт·ч)',
                                 'Производство электроэнергии на душу (кВт·ч/чел)']].copy()
        existing_regions['Тип региона'] = 'Существующий'

        all_regions = pd.concat([new_regions, existing_regions], ignore_index=True)

        numeric_cols = ['Посевные площади (тыс. га)', 'Индекс сельского хозяйства',
                       'Добыча полезных ископаемых (млн руб)', 'Индекс пром. производства',
                       'Потребление электроэнергии (млн кВт·ч)', 'Производство электроэнергии на душу (кВт·ч/чел)']

        for col in numeric_cols:
            all_regions[col] = pd.to_numeric(all_regions[col], errors='coerce')

        return all_regions.dropna(subset=numeric_cols, how='all')

    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        print("Убедитесь, что файлы находятся в правильной директории и имеют правильные имена:")
        print("- 'Данные для стат анализа регионы отдельно.xlsx'")
        print("- 'Данные для стат анализа РФ отдельно без деления на округа.xlsx'")
        return None

def calculate_comparative_metrics(df):
    """Расчет сравнительных метрик между новыми и существующими регионами"""
    if df is None:
        return {}

    metrics = {}

    new = df[df['Тип региона'] == 'Новый']
    existing = df[df['Тип региона'] == 'Существующий']

    # 1. Сравнение средних значений
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['Кластер', 'Комплексный показатель']:
            continue
        metrics[f'Среднее {col} (новые)'] = new[col].mean()
        metrics[f'Среднее {col} (существующие)'] = existing[col].mean()
        if existing[col].mean() != 0:
            metrics[f'Отношение новых к существующим ({col})'] = new[col].mean() / existing[col].mean()
        else:
            metrics[f'Отношение новых к существующим ({col})'] = np.nan

    # 2. Доля новых регионов в общих показателях
    total_new = new.select_dtypes(include=[np.number]).sum()
    total_all = df.select_dtypes(include=[np.number]).sum()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['Кластер', 'Комплексный показатель']:
            continue
        if total_all[col] != 0:
            metrics[f'Доля новых регионов в общем {col} (%)'] = (total_new[col] / total_all[col]) * 100
        else:
            metrics[f'Доля новых регионов в общем {col} (%)'] = np.nan

    # 3. Показатели концентрации
    def gini(x):
        x = np.sort(x[~np.isnan(x)])
        if len(x) == 0:
            return np.nan
        cumx = np.cumsum(x, dtype=float)
        return (len(x) + 1 - 2 * np.sum(cumx) / cumx[-1]) / len(x)

    for col in ['Добыча полезных ископаемых (млн руб)', 'Посевные площади (тыс. га)']:
        metrics[f'Коэффициент Джини {col} (все регионы)'] = gini(df[col])
        metrics[f'Коэффициент Джини {col} (новые)'] = gini(new[col])
        metrics[f'Коэффициент Джини {col} (существующие)'] = gini(existing[col])

    return metrics

def visualize_comparison(df):
    """Визуализация сравнения новых и существующих регионов"""
    if df is None:
        return

    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                   if col not in ['Кластер', 'Комплексный показатель']]

    # 1. Сравнение распределений показателей
    plt.figure(figsize=(15, 20))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(x='Тип региона', y=col, data=df)
        plt.title(f'Распределение {col}')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 2. Сравнение средних значений
    comparison = df.groupby('Тип региона')[numeric_cols].mean().T
    comparison['Отношение'] = comparison['Новый'] / comparison['Существующий']

    print("\n=== Сравнение средних значений показателей ===")
    print(tabulate(comparison, headers='keys', tablefmt='grid', floatfmt=".2f"))

    # 3. Визуализация отношений показателей
    plt.figure(figsize=(10, 6))
    comparison['Отношение'].plot(kind='bar', color='skyblue')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.title('Отношение средних значений показателей (Новые/Существующие)')
    plt.ylabel('Отношение')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. Кластерный анализ с выделением новых регионов
    X = df[numeric_cols].fillna(0)
    X = StandardScaler().fit_transform(X)

    # Оптимальное число кластеров (метод локтя)
    distortions = []
    K = range(1, min(10, len(X)+1))
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Инерция')
    plt.title('Метод локтя для определения оптимального числа кластеров')
    plt.show()

    # Кластеризация с выбранным числом кластеров
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Кластер'] = kmeans.fit_predict(X)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='Добыча полезных ископаемых (млн руб)',
        y='Потребление электроэнергии (млн кВт·ч)',
        hue='Кластер',
        style='Тип региона',
        data=df,
        palette='viridis',
        s=100
    )
    plt.title('Кластеризация регионов с выделением новых регионов')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # 5. Распределение новых регионов по кластерам
    cluster_dist = df.groupby(['Кластер', 'Тип региона']).size().unstack()
    cluster_dist['Доля новых (%)'] = (cluster_dist['Новый'] /
                                    (cluster_dist['Новый'] + cluster_dist.get('Существующий', 0))) * 100

    print("\n=== Распределение регионов по кластерам ===")
    print(tabulate(cluster_dist, headers='keys', tablefmt='grid', floatfmt=".1f"))

   # 6. Сравнение комплексного показателя
    df['Комплексный показатель'] = calculate_composite_score(df)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Тип региона', y='Комплексный показатель', data=df)
    plt.title('Сравнение комплексного показателя развития')
    plt.tight_layout()
    plt.show()

    ranked_regions = df.sort_values('Комплексный показатель', ascending=False)

    colors = []
    for region in ranked_regions['Тип региона']:
        if region == 'Новый':
            colors.append('background-color: lightgreen')
        else:
            colors.append('')

    display_df = ranked_regions[['Регион', 'Тип региона', 'Комплексный показатель'] + numeric_cols]

    print("\n=== Полный рейтинг регионов по комплексному показателю развития ===")
    print("(Новые регионы выделены цветом)\n")

    styled_df = (display_df.style
                 .apply(lambda x: colors, axis=0)
                 .format({'Комплексный показатель': "{:.2f}"})
                 .set_properties(**{'text-align': 'left'}))

    from IPython.display import display
    display(styled_df)

    plt.figure(figsize=(12, max(6, len(df) * 0.3)))  

    palette = {'Новый': 'green', 'Существующий': 'skyblue'}

    sns.barplot(y='Регион', x='Комплексный показатель',
                hue='Тип региона', data=ranked_regions,
                palette=palette, dodge=False)

    plt.title('Рейтинг регионов по комплексному показателю развития')
    plt.xlabel('Комплексный показатель')
    plt.ylabel('Регион')
    plt.legend(title='Тип региона')
    plt.tight_layout()
    plt.show()

    top_regions = ranked_regions.head(10)
    print("\n=== Топ-10 регионов по комплексному показателю ===")
    print(tabulate(top_regions[['Регион', 'Тип региона', 'Комплексный показатель']],
                  headers='keys', tablefmt='grid', floatfmt=".2f"))

def calculate_composite_score(df):
    """Расчет комплексного показателя развития региона"""
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                   if col not in ['Кластер', 'Комплексный показатель']]

    if not numeric_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)

    normalized = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Веса показателей
    weights = {
        'Посевные площади (тыс. га)': 0.15,
        'Индекс сельского хозяйства': 0.15,
        'Добыча полезных ископаемых (млн руб)': 0.25,
        'Индекс пром. производства': 0.15,
        'Потребление электроэнергии (млн кВт·ч)': 0.15,
        'Производство электроэнергии на душу (кВт·ч/чел)': 0.15
    }

    for col in normalized.columns:
        if col in weights:
            normalized[col] = normalized[col] * weights[col]

    return normalized.sum(axis=1)

def main_comparative_analysis():
    """Основная функция сравнительного анализа"""
    all_regions = load_data()

    if all_regions is None:
        return

    metrics = calculate_comparative_metrics(all_regions)

    print("\n=== Ключевые сравнительные метрики ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    visualize_comparison(all_regions)

if __name__ == "__main__":
    main_comparative_analysis()
