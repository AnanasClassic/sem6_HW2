from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT, NBEATS, PatchTST
    _HAS_NF = True
except ImportError:
    _HAS_NF = False

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def _replace_long_zeros(arr: np.ndarray, gap: int) -> np.ndarray:
    out = arr.astype(float, copy=True)
    zero_runs = np.flatnonzero(np.diff(np.concatenate(([0], out == 0, [0]))))
    starts, ends = zero_runs[::2], zero_runs[1::2]
    for s, e in zip(starts, ends):
        if (e - s) > gap:
            out[s:e] = np.nan
    return out

class ForecastPipeline:

    def __init__(
        self,
        data_path: str | Path,
        date_col: str = 'date',
        target_col: str = 'y',
        freq: str = 'D',
        test_size: int = 365,
        engine: str = 'lgbm',
        random_state: int = 42,
        store_id: str = 'STORE_1',
        zero_gap: int = 7,
    ) -> None:
        self.data_path = Path(data_path)
        self.date_col = date_col
        self.target_col = target_col
        self.freq = freq
        self.test_size = test_size
        self.engine = engine.lower()
        self.random_state = random_state
        self.store_id = store_id
        self.zero_gap = zero_gap

        self.data_: pd.DataFrame | None = None
        self.models_: Dict[str, object] = {}
        self.forecasts_: Dict[str, pd.DataFrame] = {}
        self.metrics_: pd.DataFrame | None = None

        if self.engine == 'dl' and not _HAS_NF:
            raise ImportError('engine="dl" requires neuralforecast.  pip install neuralforecast')

    def run(self) -> None:
        self._preprocess()
        self._make_horizon_targets()
        self._train_models()
        self._evaluate()
        self._plot_forecasts()

    def _preprocess(self) -> None:
        if self.data_path.is_dir():
            self._preprocess_shop_sales()
        else:
            self._preprocess_generic()

    def _preprocess_generic(self) -> None:
        if self.data_path.suffix.lower() in {'.parquet', '.pq'}:
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col).sort_index().asfreq(self.freq)

        df[self.target_col] = df[self.target_col].interpolate()

        df['dow'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        for lag in (1, 7, 30, 90):
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)

        df.dropna(inplace=True)
        self.data_ = df

    def _preprocess_shop_sales(self) -> None:
        sales_path = self.data_path / 'shop_sales.csv'
        calendar_path = self.data_path / 'shop_sales_dates.csv'

        if not (sales_path.exists() and calendar_path.exists()):
            raise FileNotFoundError('Directory must contain shop_sales.csv and shop_sales_dates.csv')

        sales = pd.read_csv(sales_path)
        calendar = pd.read_csv(calendar_path, parse_dates=['date'])

        sales = sales[sales['store_id'] == self.store_id].copy()
        sales = sales.merge(calendar[['date_id', 'date']], on='date_id', how='left')

        pivot = (
            sales.pivot_table(index='date', columns='item_id', values='cnt', aggfunc='sum')
            .sort_index()
            .asfreq('D')
        )

        for col in pivot.columns:
            pivot[col] = _replace_long_zeros(pivot[col].values, self.zero_gap)

        df = pd.DataFrame({self.target_col: pivot.sum(axis=1)})
        df[self.target_col] = df[self.target_col].interpolate()
        df = df.asfreq('D')

        df['dow'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        for lag in (1, 7, 30, 90):
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)

        df.dropna(inplace=True)
        self.data_ = df

    def _make_horizon_targets(self) -> None:
        assert self.data_ is not None
        for h, tag in zip((7, 30, 90), ('H7', 'H30', 'H90')):
            self.data_[f'y_{tag}'] = self.data_[self.target_col].shift(-h)
        self.data_.dropna(inplace=True)

    def _train_models(self) -> None:
        if self.engine == 'lgbm':
            self._train_lgbm()
        elif self.engine == 'dl':
            self._train_dl()
        else:
            raise ValueError(f'Unknown engine {self.engine}')

    def _train_lgbm(self) -> None:
        train = self.data_[:-self.test_size]
        test = self.data_[-self.test_size:]

        X_train = train.drop(columns=[c for c in train.columns if c.startswith('y_')])
        X_test = test.drop(columns=[c for c in test.columns if c.startswith('y_')])

        for y_col in ('y_H7', 'y_H30', 'y_H90'):
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=800,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
            )
            model.fit(
                X_train, train[y_col],
                eval_set=[(X_test, test[y_col])],
                eval_metric='l1',
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            self.models_[y_col] = model
            self.forecasts_[y_col] = pd.DataFrame({
                'y_true': test[y_col],
                'y_pred': model.predict(X_test, num_iteration=model.best_iteration_),
            }, index=test.index)

    def _train_dl(self) -> None:
        if not _HAS_NF:
            raise RuntimeError('neuralforecast missing')

        df_long = (
            self.data_.reset_index()
            .rename(columns={self.date_col: 'ds', self.target_col: 'y'})
            .assign(unique_id='series_0')
            [['unique_id', 'ds', 'y']]
        )

        horizons = {'H7': 7, 'H30': 30, 'H90': 90}

        for tag, h in horizons.items():
            nf = NeuralForecast(
                models=[
                    TFT(h=h, input_size=min(4 * h, 240)),
                    NBEATS(h=h, input_size=min(4 * h, 240)),
                    PatchTST(h=h, input_size=min(4 * h, 240), patch_len=16),
                ],
                freq=self.freq,
            )
            nf.fit(df_long)
            fcst = nf.predict().set_index('ds')
            y_hat = (
                fcst[[c for c in fcst.columns if c.endswith('_mean')]]
                .mean(axis=1)
                .rename('y_pred')
            )
            y_true = (
                self.data_[self.data_.index >= y_hat.index[0]][self.target_col]
                .iloc[: len(y_hat)]
            ).rename('y_true')
            self.forecasts_[f'y_{tag}'] = pd.concat([y_true, y_hat], axis=1)
            self.models_[f'DL_{tag}'] = nf

    def _evaluate(self) -> None:
        rows: list[dict[str, float]] = []

        for tag, df_pred in self.forecasts_.items():
            df_clean = df_pred.dropna()

            if df_clean.empty:
                print(f'⚠️  {tag}: no valid predictions, skipped')
                continue

            mae  = mean_absolute_error(df_clean.y_true, df_clean.y_pred)
            rmse = np.sqrt(mean_squared_error(df_clean.y_true, df_clean.y_pred))
            mape = np.mean(np.abs((df_clean.y_true - df_clean.y_pred) /
                                df_clean.y_true)) * 100

            rows.append({"horizon": tag,
                        "MAE": mae, "RMSE": rmse, "MAPE_%": mape})

        if not rows:
            print("❌  Evaluation skipped: no valid forecasts")
            self.metrics_ = pd.DataFrame()
            return

        self.metrics_ = (
            pd.DataFrame(rows)
            .set_index("horizon")
            .round(3)
        )
        print("\n=== Hold-out metrics ===")
        print(self.metrics_)

    def _plot_forecasts(self) -> None:
        for tag, df_pred in self.forecasts_.items():
            df_pred[['y_true', 'y_pred']].plot(
                figsize=(10, 4), title=f'{tag}: actual vs forecast')
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Forecast CLI')
    p.add_argument('data', help='Path to CSV/Parquet or directory with shop_sales* files')
    p.add_argument('--engine', choices=['lgbm', 'dl'], default='lgbm')
    p.add_argument('--test', type=int, default=365)
    p.add_argument('--store_id', default='STORE_1')
    p.add_argument('--zero_gap', type=int, default=7)
    args = p.parse_args()

    pipeline = ForecastPipeline(
        data_path=args.data,
        engine=args.engine,
        test_size=args.test,
        store_id=args.store_id,
        zero_gap=args.zero_gap,
    )
    pipeline.run()
