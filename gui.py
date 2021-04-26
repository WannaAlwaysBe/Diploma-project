import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from adtk.data import validate_series
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.detector import OutlierDetector

from sklearn.neighbors import LocalOutlierFactor

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import Menu, filedialog
from tkinter import scrolledtext
from tkinter import simpledialog

from math import sqrt


root = tk.Tk()
root.title("Diploma")


def get_file():
    """Отримуємо файл EXCEL"""
    file = filedialog.askopenfilename()

    global dataset
    global resampled_dataset
    dataset = pd.read_excel(file, index_col='Date')
    dataset = validate_series(dataset)
    resampled_dataset = dataset['Quantity'].resample('MS').sum()

def show_data():
    """Відображаємо дані"""
    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, dataset)
    txt.configure(state=tk.DISABLED)

def show_data_plot():
    """Відображаємо графік"""
    dataset.plot(figsize=(15,6))
    plt.show()


def show_anomaly_describe():
    """Функція describe для даних"""
    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, dataset.describe())
    txt.configure(state=tk.DISABLED)

def show_thresholdad_plot():
    """Фунцкція tresholdad для пошуку аномалій"""
    threshold = simpledialog.askfloat(title="ThresholdAD", prompt="Input threshold: ")
    threshold_ad = ThresholdAD(high=threshold)
    anomalies = threshold_ad.detect(dataset)
    plot(dataset, anomaly=anomalies, ts_linewidth=1, anomaly_markersize=5, anomaly_color='red', anomaly_tag = 'marker')
    
    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, anomalies[anomalies.Quantity])
    txt.configure(state=tk.DISABLED)
    
    plt.show()

def show_outlierdetector_plot():
    """Функція outlierdetector для пошуку аномалій"""
    contamination = simpledialog.askfloat(title="OutlierDetector", prompt="Input contamination: ")
    outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=contamination))
    anomalies = outlier_detector.fit_detect(dataset)
    plot(dataset, anomaly=anomalies, ts_linewidth=2, anomaly_color='red', anomaly_alpha=0.3, curve_group='all')

    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, anomalies[anomalies])
    txt.configure(state=tk.DISABLED)

    plt.show()

def show_seasonal_plot():
    """Відображаємо графік сезонності"""
    plt.figure(figsize=(19,8), dpi= 80)
    for i, y in enumerate(dataset.index.year.unique()):
        plt.plot(list(range(1,len(dataset[dataset.index.year==y])+1)), dataset[dataset.index.year==y][dataset.columns[0]].values, label=y)            
    plt.title("Сезонность по периодам")
    plt.legend(loc="best")
    plt.show()


def show_decomposition_mul():
    """Мультиплікативна декомпозиція"""
    result_M = seasonal_decompose(dataset.Quantity, model='multiplicative', freq=12)

    plt.rcParams.update({'figure.figsize': (10,10)})
    result_M.plot().suptitle('Multiplicative model')
    plt.show()

def show_decomposition_add():
    """Адитивна декомпозиція"""
    result_M = seasonal_decompose(dataset.Quantity, model='additive', freq=12)

    plt.rcParams.update({'figure.figsize': (10,10)})
    result_M.plot().suptitle('Additive model')
    plt.show()


def make_tt_datasets():
    """Розбиваємо на тренувальні та тестові дані"""
    global train
    global test
    global start_train
    global start_test
    global end_train
    start_train = simpledialog.askstring(title="Train", prompt="Start of train dataset(year): ")
    end_train = simpledialog.askstring(title="Train", prompt="End of train dataset(year): ")
    start_test = simpledialog.askstring(title="Test", prompt="Start of test dataset(year): ")
    train=dataset[start_train:end_train]
    test=dataset[start_test]

def show_tt_datasets():
    """Відображення тренувальних та тестових даних"""
    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, 'Train dataset')
    txt.insert(tk.INSERT, train)
    txt.insert(tk.INSERT, '\n Test dataset')
    txt.insert(tk.INSERT, test)
    txt.configure(state=tk.DISABLED)


def metrics(real, forecast):
    """Функція для відображення метрики для тесту Дікі-Фуллера"""
    if type(real)==pd.core.frame.DataFrame:
        real=real[real.columns[0]].values
    
    txt.insert(tk.INSERT, "Тест на стационарность:\n")
    dftest = adfuller(real-forecast, autolag='AIC')
    txt.insert(tk.INSERT, "\tT-статистика = {:.3f}\n".format(dftest[0]))
    txt.insert(tk.INSERT, "\tP-значение = {:.3f}\n".format(dftest[1]))
    txt.insert(tk.INSERT, "Критические значения:\n")
    for k, v in dftest[4].items():
        txt.insert(tk.INSERT, "\t{}: {} - Данные {} стационарны с вероятностью {}% процентов\n".format(k, v, "не" if v<dftest[0] else "", 100-int(k[:-1])))
    
    forecast=np.array(forecast)
    MSE = np.square(np.subtract(real,forecast)).mean()
    RMSE = sqrt(MSE)
    txt.insert(tk.INSERT, "MSE: {}\n".format(MSE))
    txt.insert(tk.INSERT, "RMSE: {}\n".format(RMSE))


def configure_holt_model():
    """Створення моделі Хольта-Вінтерса"""
    fit1 = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add').fit()
    fit2 = ExponentialSmoothing(train, seasonal_periods=12, trend='mul', seasonal='mul').fit()
    fit3 = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='mul').fit()
    fit4 = ExponentialSmoothing(train, seasonal_periods=12, trend='mul', seasonal='add').fit()

    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    show_config_holt_model(fit1, 'add', 'add')
    show_config_holt_model(fit2, 'mul', 'mul')
    show_config_holt_model(fit3, 'add', 'mul')
    show_config_holt_model(fit4, 'mul', 'add')
    txt.configure(state=tk.DISABLED)

def show_config_holt_model(model, season, trend):
    """Відображення метрик"""
    txt.insert(tk.INSERT, '='*40+'\n')
    txt.insert(tk.INSERT, 'train, season={}, trend={}\n'.format(season, trend))
    metrics(train, model.fittedvalues)
    txt.insert(tk.INSERT, '\ntest, season={}, trend={}\n'.format(season, trend))
    metrics(test, model.forecast(len(test)))

def run_holt_model():
    """Прогноз Холт-Вінтерс"""
    trend = simpledialog.askstring(title="Trend", prompt="Trend (add/mul): ")
    seasonal = simpledialog.askstring(title="Seasonal", prompt="Seasonal (add/mul): ")
    fit_holt = ExponentialSmoothing(dataset, seasonal_periods=12, trend=trend, seasonal=seasonal).fit()
    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, 'dataset, season={}, trend={}\n'.format(seasonal, trend))
    txt.insert(tk.INSERT, '='*40+'\n')
    txt.insert(tk.INSERT, 'Forecast for 12 months\n')
    txt.insert(tk.INSERT, fit_holt.forecast(12))
    txt.configure(state=tk.DISABLED)

    ax = dataset.plot(figsize=(15,6), color='black', title="Прогноз методом Хольта Винтерса" )
    fit_holt.fittedvalues.plot(ax=ax, style='--', color='red')
    fit_holt.forecast(12).plot(ax=ax, style='--', color='green')
    plt.show()

def show_autocorrelation_plot():
    """Відображення графіків автокореляції"""
    plt.rcParams.update({'figure.figsize': (10,3)})
    plot_acf(train.Quantity, lags=12)
    plot_pacf(train.Quantity, lags=12)
    plt.show()

def find_best_arima():
    """Знаходження найкращої моделі ARIMA"""
    model = auto_arima(dataset, seasonal=True, m=12, trace=True, suppress_warnings=True, error_action='ignore', stepwise=True)
    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, model)
    txt.configure(state=tk.DISABLED)

def train_arima():
    """Тренування ARIMA"""
    ord = simpledialog.askstring(title="order", prompt="Trend (p d q): ")
    sord = simpledialog.askstring(title="seasonal order", prompt="Seasonal (P D Q): ")
    ord = [int(i) for i in ord.split(' ')]
    sord = [int(i) for i in sord.split(' ')]

    mod = sm.tsa.statespace.SARIMAX(dataset,
                                order=(ord[0], ord[1], ord[2]),
                                seasonal_order=(sord[0], sord[1], sord[2], 12),
                                enforce_invertibility=False)

    global fit_arima
    fit_arima = mod.fit(disp=0)
    pred_start = '{}-01-01'.format(start_test)
    pred = fit_arima.get_prediction(start=pd.to_datetime(pred_start), dynamic=False)

    dataset_forecasted = pred.predicted_mean
    dataset_truth = resampled_dataset['{}-01-01'.format(start_train):]

    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, dataset_forecasted)
    MSE = np.square(np.subtract(dataset_truth,dataset_forecasted)).mean()
    RMSE = sqrt(MSE)
    txt.insert(tk.INSERT, "MSE: {}\n".format(MSE))
    txt.insert(tk.INSERT, "RMSE: {}\n".format(RMSE))
    txt.configure(state=tk.DISABLED)

    pred_ci = pred.conf_int()
    ax = dataset[start_train:].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    plt.legend()
    plt.show()

def diagnostics_arima():
    """Графіки діагностики ARIMA"""
    fit_arima.plot_diagnostics(figsize=(16, 8))
    plt.show()

def run_arima():
    """Запуск прогнозу ARIMA"""
    pred_uc = fit_arima.get_forecast(steps=12)
    pred_ci = pred_uc.conf_int()

    txt.configure(state=tk.NORMAL)
    txt.delete('1.0', tk.END)
    txt.insert(tk.INSERT, '\n'+'='*40+'\n')
    txt.insert(tk.INSERT, fit_arima.forecast(12))
    txt.configure(state=tk.DISABLED)

    ax = dataset[start_test:].plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    plt.legend()
    plt.show()


txt = scrolledtext.ScrolledText(root, width=40, height=20, state=tk.DISABLED)
txt.grid(columnspan=5, rowspan=7, column=0, row=0)

menu = Menu(root)

data = Menu(menu, tearoff=0)
anomaly = Menu(menu, tearoff=0)
decomposition = Menu(menu, tearoff=0)
datasets = Menu(menu, tearoff=0)
models = Menu(menu, tearoff=0)

data.add_command(label='Download', command=get_file)
data.add_command(label='Show data', command=show_data)
data.add_command(label='Show plot', command=show_data_plot)

anomaly.add_command(label='Show describe', command=show_anomaly_describe)
anomaly.add_command(label='TresholdAD', command=show_thresholdad_plot)
anomaly.add_command(label='OutlierDetector', command=show_outlierdetector_plot)
anomaly.add_command(label='Seasonal', command=show_seasonal_plot)

decomposition.add_command(label='Multiplicative', command=show_decomposition_mul)
decomposition.add_command(label='Additive', command=show_decomposition_add)

datasets.add_command(label='Beat dataset', command=make_tt_datasets)
datasets.add_command(label='Show datasets', command=show_tt_datasets)

models.add_command(label='Test Holt-Winters', command=configure_holt_model)
models.add_command(label='Run Holt-Winters', command=run_holt_model)
models.add_command(label='Autocorrelation', command=show_autocorrelation_plot)
models.add_command(label='Find best ARIMA', command=find_best_arima)
models.add_command(label='Train your ARIMA', command=train_arima)
models.add_command(label='Diagnostics your ARIMA', command=diagnostics_arima)
models.add_command(label='Run your ARIMA', command=run_arima)

menu.add_cascade(label='Data', menu=data)
menu.add_cascade(label='Anomaly', menu=anomaly)
menu.add_cascade(label='Decomposition', menu=decomposition)
menu.add_cascade(label='Datasets', menu=datasets)
menu.add_cascade(label='Model', menu=models)


root.config(menu=menu)
root.mainloop()