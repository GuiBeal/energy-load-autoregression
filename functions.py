import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import math
import numpy as np
import pandas as pd

from statistics import mean, stdev
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error

matplotlib.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  "pgf.texsystem": "pdflatex",
  "pgf.rcfonts": False,
})

format = "eps"
extension = ".eps"

colormap = plt.get_cmap("tab10")

def plot_data(time, data, figsize=(8,4), xlabel="Tempo", ylabel="Carga [MWmed/dia]", linewidth=1, saveFileName=""):
  plt.figure(figsize=figsize)
  ax = plt.gca()
  plt.plot(time, data, linewidth=linewidth)
  plt.xlim(time[0], time[-1])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xticks(rotation=45)
  ax.xaxis.set_major_locator(mdates.YearLocator(1))
  ax.xaxis.set_minor_locator(mdates.MonthLocator(range(1,13,3)))
  plt.grid(which="major")
  plt.grid(which="minor", linestyle=":")
  plt.tight_layout()

  if saveFileName:
    plt.savefig(saveFileName+extension, format=format)

  plt.show()

def plot_rolling(time, data, window=12, figsize=(8,7), xlabel="Tempo", ylabel="Carga [MWmed/dia]", linewidth=1, saveFileName=""):
  s = pd.Series(data)
  mean = s.rolling(window).mean()
  std = s.rolling(window).std()

  (fig, axs) = plt.subplots(2, figsize=figsize, sharex=True)
  axs[0].plot(time, data, label="Série", color=colormap(0), linewidth=linewidth)
  axs[0].plot(time, mean, label="Média Móvel", color=colormap(1), linewidth=linewidth)
  axs[1].plot(time, std, label="Desvio Padrão Móvel", color=colormap(2), linewidth=linewidth)
  for ax in axs:
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel(ylabel)
    ax.xaxis.set_tick_params(labelrotation=45)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(range(1,13,3)))
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.legend()
  axs[1].set_xlabel(xlabel)
  plt.tight_layout()

  if saveFileName:
    plt.savefig(saveFileName+extension, format=format)

  plt.show()

def plot_params(theta, figsize=(8,4), constant=True, saveFileName=""):
  plt.figure(figsize=figsize)
  (markers, stemlines, baseline) = plt.stem(theta)
  ax = plt.gca()
  plt.setp(stemlines, linestyle="-.", color="gray", linewidth=0.5)
  plt.setp(baseline, visible=False)
  plt.xlabel(r"$\theta_i$")
  ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
  ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
  if constant:
    print("Constant: ", theta[0])
    plt.ylim((1.1*min(theta[1:]),1.1*max(theta[1:])))
  plt.grid(which="major")
  plt.grid(which="minor", linestyle=":")
  plt.tight_layout()

  if saveFileName:
    plt.savefig(saveFileName+extension, format=format)

  plt.show()

def plot_test(time_data, data, time_forecast, forecast, figsize=(8,4), xlabel="Tempo", ylabel="Carga [MWmed/dia]", linewidth=1, saveFileName=""):
  plt.figure(figsize=figsize)
  ax = plt.gca()
  plt.plot(time_data, data, marker=".", label="Real", linewidth=linewidth)
  plt.plot(time_forecast, forecast, marker=".", label="Previsão", linewidth=linewidth)
  plt.xlim(min(time_data[0], time_forecast[0]), max(time_data[-1], time_forecast[-1]))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xticks(rotation=45)
  ax.xaxis.set_major_locator(mdates.YearLocator(1))
  ax.xaxis.set_minor_locator(mdates.MonthLocator(range(1,13,3)))
  plt.grid(which="major")
  plt.grid(which="minor", linestyle=":")
  plt.legend()
  plt.tight_layout()

  if saveFileName:
    plt.savefig(saveFileName+extension, format=format)

  plt.show()

def plot_partial_autocorrelation(data, lags, alpha=0.05, figsize=(8,4), saveFileName=""):
  plt.figure(figsize=figsize)
  ax = plt.gca()
  plot_pacf(data, lags=lags, method="ywm", alpha=alpha, ax=ax)
  plt.title(None)
  plt.xlim(0, lags)
  plt.xlabel("Atraso")
  plt.ylabel("Autocorrelação Parcial")
  plt.xticks(np.arange(0,lags+1,6))
  plt.grid()
  plt.tight_layout()

  if saveFileName:
    ax.set_rasterized(True) # overcomes transparency issues
    plt.savefig(saveFileName+extension, format=format, dpi=400)

  plt.show()

def adf_test(data):
  result = adfuller(data)
  stats = result[0]
  p_value = result[1]
  critical_values = result[4]
  print("ADF Statistic: ", stats)
  print("P-Value: ", p_value)
  print("Critical Values:")
  for threshold, adf_stat in critical_values.items():
    print("\t%s: %.2f" % (threshold, adf_stat))
  print("Stationary") if p_value < 0.05 else print("Not Stationary")

def kpss_test(data, regression="ct"):
  result = kpss(data, regression=regression)
  stats = result[0]
  p_value = result[1]
  critical_values = result[3]
  print("KPSS Statistic: ", stats)
  print("P-Value: ", p_value)
  print("Critical Values:")
  for threshold, kpss_stat in critical_values.items():
    print("\t%s: %.2f" % (threshold, kpss_stat))
  print("Stationary") if p_value >= 0.05 else print("Not Stationary")

def autoregression_batch(data, n, constant=True):
  N = len(data)

  T = toeplitz(data)[n-1:N-1,:n]
  Phi = (np.hstack([np.ones((N-n,1)), T]) if constant else T)

  y = data[n:]

  # theta = np.linalg.inv(Phi.transpose() @ Phi) @ Phi.transpose() @ y
  theta = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ y)

  p = apply_autorregresion(data, theta, constant)
  mse = mean_squared_error(data[n:], p)

  return theta, mse

def autoregression_batch_sum(data, n, constant=True):
  N = len(data)
  m = (n+1 if constant else n)

  P_inv = np.zeros((m,m))
  v = np.zeros((m))
  for i in range(n,N):
    phi = np.flip(data[i-n:i])
    phi = (np.hstack([1, phi]) if constant else phi)

    P_inv = P_inv + np.outer(phi, phi)
    v = v + data[i]*phi

  P = np.linalg.inv(P_inv)
  theta = P @ v

  p = apply_autorregresion(data, theta, constant)
  mse = mean_squared_error(data[n:], p)

  return theta, mse, P

def autoregression_recursive(data, n, theta=None, P=None, constant=True):
  N = len(data)
  m = (n+1 if constant else n)

  if theta is None:
    theta = np.zeros((m))

  if P is None:
    P = np.eye(m,m)

  for i in range(n,N):
    phi = np.flip(data[i-n:i])
    phi = (np.hstack([1, phi]) if constant else phi)

    y = phi @ theta
    e = data[i] - y

    k = (1/(1+phi @ P @ phi))*(P @ phi)

    theta = theta + k * e
    P = P - np.outer(k, phi) @ P

  p = apply_autorregresion(data, theta, constant)
  mse = mean_squared_error(data[n:], p)

  return theta, mse, P

def apply_autorregresion(data, theta, constant=True):
 n = len(theta) - (1 if constant else 0)
 T = toeplitz(data)[n-1:-1,:n]
 Phi = (np.hstack([np.ones((len(T),1)), T]) if constant else T)
 y = Phi @ theta
 return y

def mse_backsteps(data, n_range=range(1,80), train_set=0.8, figsize=(8,7), saveFileName=""):
  N = len(data)
  m = math.floor(train_set*N)

  train = data[:m]

  mses_train = []
  mses_test  = []
  for n in n_range:
    test = data[m-n:]

    theta, mse_train = autoregression_batch(train, n)
    y = apply_autorregresion(test, theta)

    mse_test = mean_squared_error(test[n:], y)

    mses_train.append(mse_train)
    mses_test.append(mse_test)

  (fig, axs) = plt.subplots(2, figsize=figsize, sharex=True)
  axs[0].plot(n_range, mses_train, label="Treinamento")
  axs[1].plot(n_range, mses_test, label="Validação")
  for ax in axs:
    ax.set_xlim(n_range[0], n_range[-1])
    ax.set_ylabel("Erro Quadrático Médio")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.legend()
  axs[1].set_xlabel("n")
  plt.tight_layout()

  if saveFileName:
    plt.savefig(saveFileName+extension, format=format)

  plt.show()

  return mses_test

def k_fold_cross_validation(t, data, n, k, constant=True, xlabel="Tempo", ylabel="Carga [MWmed/dia]", figsize=(8,4), saveFileName=["", "", ""]):
  T = toeplitz(data)[n-1:-1,:n]
  Phi = (np.hstack([np.ones((len(T),1)), T]) if constant else T)

  y = data[n:]
  t_ = t[n:]

  Phi_folded = np.array_split(Phi, k)
  y_folded   = np.array_split(y  , k)
  t_folded   = np.array_split(t_ , k)

  thetas = []
  preds_test  = []
  mses_test   = []
  preds_train = []
  mses_train  = []
  errors = []
  ts     = []
  for i in range(k):
    Phi_test = Phi_folded[i]
    y_test = y_folded[i]
    t_test = t_folded[i]

    Phi_train = Phi_folded[:i] + Phi_folded[i+1:]
    y_train = y_folded[:i] + y_folded[i+1:]

    Phi_train = np.array(np.concatenate(Phi_train))
    y_train = np.concatenate(y_train)

    # theta = np.linalg.inv(Phi_train.transpose() @ Phi_train) @ Phi_train.transpose() @ y_train
    theta = np.linalg.solve(Phi_train.transpose() @ Phi_train, Phi_train.transpose() @ y_train)
    thetas.append(theta)

    pred_train = Phi_train @ theta
    preds_train.append(pred_train)

    pred_test = Phi_test @ theta
    preds_test.append(pred_test)

    mse_train = mean_squared_error(pred_train, y_train)
    mses_train.append(mse_train)

    mse_test = mean_squared_error(pred_test, y_test)
    mses_test.append(mse_test)

    error = y_test - pred_test
    errors.append(error)

    ts.append(t_test)

  plt.figure(figsize=figsize)
  ax = plt.gca()
  theta_max = 0
  theta_min = 0
  if constant:
    print("Constants:")
  for i in range(k):
    print(f"Fold {i+1}: {thetas[i][0]}")
    (markers, stemlines, baseline) = plt.stem(thetas[i], label=i+1)
    plt.setp(markers, color=colormap(i+1))
    plt.setp(stemlines, linestyle="-.", color="gray", linewidth=0.5)
    plt.setp(baseline, visible=False)
    theta_max = max(theta_max, max((thetas[i][1:] if constant else thetas[i])))
    theta_min = min(theta_min, min((thetas[i][1:] if constant else thetas[i])))
  if constant:
    plt.ylim(1.1*theta_min, 1.1*theta_max)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
  ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
  plt.xlabel(r"$\theta_i$")
  plt.legend(title="Divisão", ncol=k)
  plt.grid(which="major")
  plt.grid(which="minor", linestyle=":")
  plt.tight_layout()

  if saveFileName[0]:
    plt.savefig(saveFileName[0]+extension, format=format)

  plt.show()

  plt.figure(figsize=figsize)
  plt.bar(np.linspace(1,k,k,dtype=int), mses_train, color=colormap.colors[1:])
  for i in range(k):
    plt.text(i+1, mses_train[i]//2, f"{mses_train[i]:.4g}", ha="center")
  plt.xlabel("Divisão")
  plt.ylabel("MSE")
  plt.grid()
  plt.tight_layout()

  if saveFileName[1]:
    plt.savefig(saveFileName[1]+extension, format=format)

  plt.show()

  plt.figure(figsize=figsize)
  plt.bar(np.linspace(1,k,k,dtype=int), mses_test, color=colormap.colors[1:])
  for i in range(k):
    plt.text(i+1, mses_test[i]//2, f"{mses_test[i]:.4g}", ha="center")
  plt.xlabel("Divisão")
  plt.ylabel("MSE")
  plt.grid()
  plt.tight_layout()

  if saveFileName[2]:
    plt.savefig(saveFileName[2]+extension, format=format)

  plt.show()

  plt.figure(figsize=figsize)
  ax = plt.gca()
  plt.plot(t, data, marker=".")
  for i in range(k):
    plt.plot(ts[i], preds_test[i], marker=".", color=colormap(i+1), label=i+1)
  plt.xlim(t[0], t[-1])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xticks(rotation=45)
  ax.xaxis.set_major_locator(mdates.YearLocator(1))
  ax.xaxis.set_minor_locator(mdates.MonthLocator(range(1,13,3)))
  plt.legend(title="Divisão", ncol=k)
  plt.grid(which="major")
  plt.grid(which="minor", linestyle=":")
  plt.tight_layout()

  if saveFileName[3]:
    plt.savefig(saveFileName[3]+extension, format=format)

  plt.show()

  return thetas, ts, preds_test, mses_test, preds_train, mses_train
