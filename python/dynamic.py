"""Динамика РФ отдельно по округам 2020-2022"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tabulate import tabulate
from scipy.stats import linregress

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

    indicators = {
        'Посевные площади': ['2020', '2021', '2022'],
        'Индекс сельского хозяйства': ['2020.1', '2021.1', '2022.1'],
        'Добыча полезных ископаемых': ['2020.2', '2021.2', '2022.2'],
        'Индекс пром. производства': ['2020.3', '2021.3', '2022.3'],
        'Потребление электроэнергии': ['2020.4', '2021.4', '2022.4'],
        'Производство электроэнергии на душу': ['2020.5', '2021.5', '2022.5']
    }

    years = [2020, 2021, 2022]
    dfs = []

    for i, year in enumerate(years):
        temp_df = analysis_data[['НАИМЕНОВАНИЕ РЕГИОНА']].copy()
        temp_df.columns = ['Регион']
        temp_df['Год'] = year

        for indicator, cols in indicators.items():
            if cols[i] in analysis_data.columns:
                temp_df[indicator] = pd.to_numeric(analysis_data[cols[i]], errors='coerce')

        dfs.append(temp_df)

    long_data = pd.concat(dfs)
    return long_data.dropna(subset=indicators.keys(), how='all')

def calculate_trend_metrics(data):
    """Расчет метрик динамики для каждого показателя и региона"""
    metrics = {}
    regions = data['Регион'].unique()
    indicators = ['Посевные площади', 'Индекс сельского хозяйства',
                 'Добыча полезных ископаемых', 'Индекс пром. производства',
                 'Потребление электроэнергии', 'Производство электроэнергии на душу']

    for region in regions:
        region_data = data[data['Регион'] == region].sort_values('Год')
        region_metrics = {}

        for indicator in indicators:
            if indicator not in region_data.columns:
                continue

            values = pd.to_numeric(region_data[indicator], errors='coerce').dropna().values
            years = region_data['Год'].values

            if len(values) > 1 and len(values) == len(years):

                    abs_growth = values[-1] - values[0]

                    growth_rate = (values[-1] / values[0] * 100) if values[0] != 0 else np.nan

                    n_years = years[-1] - years[0]
                    if values[0] != 0 and n_years > 0:
                        cagr = (values[-1] / values[0]) ** (1/n_years) * 100
                    else:
                        cagr = np.nan

                    slope, _, _, _, _ = linregress(years, values)

                    region_metrics[f'{indicator} - Абсолютный прирост'] = abs_growth
                    region_metrics[f'{indicator} - Темп роста (%)'] = growth_rate
                    region_metrics[f'{indicator} - CAGR (%)'] = cagr
                    region_metrics[f'{indicator} - Тренд (наклон)'] = slope
                except Exception as e:
                    print(f"Ошибка при расчете метрик для {region}, {indicator}: {str(e)}")
                    continue

        metrics[region] = region_metrics

    return metrics

def calculate_volatility(data):
    """Расчет волатильности показателей по регионам"""
    volatility = {}
    indicators = ['Посевные площади', 'Индекс сельского хозяйства',
                 'Добыча полезных ископаемых', 'Индекс пром. производства',
                 'Потребление электроэнергии', 'Производство электроэнергии на душу']

    for indicator in indicators:
        if indicator in data.columns:
            vol = data.groupby('Регион')[indicator].std() / data.groupby('Регион')[indicator].mean() * 100
            volatility[f'{indicator} - Волатильность (%)'] = vol.to_dict()

    return volatility

def visualize_dynamics(data, indicator):
    """Визуализация динамики показателя по регионам"""
    if indicator not in data.columns:
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Год', y=indicator, hue='Регион', marker='o')
    plt.title(f'Динамика показателя "{indicator}" по федеральным округам')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_trends(trend_metrics, indicator):
    """Визуализация трендовых метрик по регионам"""
    metrics_df = []
    for region, metrics in trend_metrics.items():
        if f'{indicator} - Тренд (наклон)' in metrics:
            metrics_df.append({
                'Регион': region,
                'Наклон тренда': metrics[f'{indicator} - Тренд (наклон)'],
                'Темп роста (%)': metrics[f'{indicator} - Темп роста (%)'] if f'{indicator} - Темп роста (%)' in metrics else np.nan,
                'CAGR (%)': metrics[f'{indicator} - CAGR (%)'] if f'{indicator} - CAGR (%)' in metrics else np.nan
            })

    if not metrics_df:
        return

    metrics_df = pd.DataFrame(metrics_df)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=metrics_df, x='Наклон тренда', y='Темп роста (%)', hue='Регион', s=100)
    plt.title(f'Сравнение трендовых метрик для "{indicator}"')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.axhline(y=100, color='gray', linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main_dynamic_analysis():
    """Основная функция анализа динамики"""
    rf_data = load_rf_data()
    long_data = prepare_rf_data(rf_data)

    print("\n=== Анализ динамики показателей ===")

    # 1. Метрики трендов для каждого региона
    trend_metrics = calculate_trend_metrics(long_data)

    # 2. Волатильность показателей
    volatility = calculate_volatility(long_data)

    indicators = ['Посевные площади', 'Индекс сельского хозяйства',
                 'Добыча полезных ископаемых', 'Индекс пром. производства',
                 'Потребление электроэнергии', 'Производство электроэнергии на душу']

    for indicator in indicators:
        # Динамика показателя
        visualize_dynamics(long_data, indicator)

        # Трендовые метрики
        visualize_trends(trend_metrics, indicator)

    print("\n=== Основные трендовые метрики ===")
    for region, metrics in trend_metrics.items():
        print(f"\nРегион: {region}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")

    print("\n=== Волатильность показателей ===")
    for metric, values in volatility.items():
        print(f"\n{metric}:")
        for region, value in values.items():
            print(f"  {region}: {value:.2f}%")

if __name__ == "__main__":
    main_dynamic_analysis()
