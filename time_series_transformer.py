from tqdm import tqdm
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare(src='data/weatherHistory.csv', dst='data/weatherHistory_sorted.csv'):
    """One-time cleanup: sort chronologically, drop duplicate timestamps."""
    df = pd.read_csv(src)
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df = df.sort_values('Formatted Date')

    dups = df['Formatted Date'].duplicated().sum()
    df = df.drop_duplicates('Formatted Date', keep='first')

    df.to_csv(dst, index=False)

    d = df['Formatted Date']
    gaps = d.diff()[d.diff() > pd.Timedelta('1h')]
    print(f'{len(df)} rows written, monotonic: {d.is_monotonic_increasing}')
    print(f'{dups} duplicate timestamps dropped, {len(gaps)} gaps >1h')


class WindowGenerator:

    def __init__(self, csv_path, columns, index_column, window_size, horizon=1, chunksize=5000, f_split=0.9):
        self._csv_path = Path(csv_path)
        self._columns = columns
        self._index_column = index_column
        self._chunksize = chunksize
        self._window_size = window_size
        self._horizon = horizon
        self._f_split = f_split

        header, chunk_offsets_train, chunk_offsets_test, nrows, n_train_rows = (
            self._build_chunk_offsets(
                csv_path=self._csv_path, 
                chunksize=self._chunksize, 
                f_split=self._f_split,
            )
        )
        self._header = header
        self.nrows = nrows
        self.n_train_rows = n_train_rows
        self._chunk_offsets_train = chunk_offsets_train
        self._chunk_offsets_test = chunk_offsets_test

        # feature statistics from training rows only — never touches test data
        self._mean, self._std = self._compute_train_stats()
        self.target_mean = float(self._mean[0])
        self.target_std = float(self._std[0])

    @staticmethod
    def _build_chunk_offsets(csv_path, chunksize, f_split):
        """One sequential pass: record byte offset at the start of each chunk."""
        offsets = []
        nrows = 0
        with open(csv_path, 'rb') as f:
            header = f.readline().decode().strip().split(',')  # read header row
            while True:  # loop until we're done
                pos = f.tell()  # capture the file cursor's current byte position
                line = f.readline()  # advance cursor to peek into chunk and check we're not at end of file
                if not line:
                    break
                nrows += 1
                offsets.append(pos)
                for _ in range(chunksize - 1):  # loop until end of current chunk, breaking if we reach end of file
                    if not f.readline():
                        break
                    nrows += 1

        split_row = int(f_split * nrows)
        split_chunk_index = split_row // chunksize
        offsets_train = offsets[:split_chunk_index]
        offsets_test = offsets[split_chunk_index:]
        n_train_rows = len(offsets_train) * chunksize

        return header, offsets_train, offsets_test, nrows, n_train_rows

    @staticmethod
    def _read_chunk(csv_path, offset, header, columns, index_column, chunksize):
        """Seek to a byte offset and read nrows rows. Returns (values, timestamps)."""
        with open(csv_path) as f:
            f.seek(offset)  # find starting position in file
            chunk = pd.read_csv(
                f,
                header=None,
                names=header,
                usecols=columns,
                parse_dates=[index_column],
                index_col=index_column,
                nrows=chunksize,
            )
        return chunk.values.astype('float32'), chunk.index.values

    def _compute_train_stats(self):
        """Streaming mean/std over training chunks only."""
        total = total_sq = None
        n = 0
        for offset in self._chunk_offsets_train:
            arr, _ = self._read_chunk(
                self._csv_path, 
                offset, 
                self._header, 
                self._columns,
                self._index_column, 
                self._chunksize,
            )
            a64 = arr.astype('float64')
            if total is None:
                total, total_sq = a64.sum(axis=0), (a64 ** 2).sum(axis=0)
            else:
                total += a64.sum(axis=0)
                total_sq += (a64 ** 2).sum(axis=0)
            n += len(arr)

        mean = total / n
        std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-8))
        return mean.astype('float32'), std.astype('float32')

    def _build_windows_from_offset(self, offset, shuffle):
        # read chunk including enough lookahead rows to use the bottom rows of the chunk
        arr, idx = self._read_chunk(
            self._csv_path, 
            offset, 
            self._header, 
            self._columns,
            self._index_column, 
            self._chunksize + self._window_size + self._horizon
        )

        arr = (arr - self._mean) / self._std

        n_starts = len(arr) - self._window_size - self._horizon  # number of valid window starting positions
        # permute starting positions for training, keep ordered for eval
        starts = np.random.permutation(n_starts) if shuffle else range(n_starts)
        for i in starts:
            X = arr[i:i + self._window_size]
            y = arr[(i + self._window_size):(i + self._window_size + self._horizon), 0]
            t = idx[i + self._window_size]
            yield torch.from_numpy(X.copy()), torch.from_numpy(y.copy()), t

    def get_random_windows(self):
        offsets = self._chunk_offsets_train.copy()
        random.shuffle(offsets)
        for offset in offsets:
            for window in self._build_windows_from_offset(offset, shuffle=True):
                yield window

    def get_chronological_windows(self):
        for offset in self._chunk_offsets_test:
            for window in self._build_windows_from_offset(offset, shuffle=False):
                yield window


class ForecastTransformer(nn.Module):

    def __init__(self, horizon=1):
        super().__init__()
        d_model = 6
        self.horizon = horizon
        self.embedding = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=16,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )
        self.output = nn.Linear(d_model, horizon)

    def forward(self, x):

        # input: (batch, window, 5), output: (batch, window, d_model)
        x = self.embedding(x)

        # input: (batch, window, d_model), output from self-attention: (batch, window, d_model)
        x = self.encoder(x)

        # keep only the last timestep: (batch, d_model); attention makes each row a weighted blend of all 
        # timesteps' value-vectors — that's exactly what softmax(QKᵀ/√d_k) @ V does
        x = x[:, -1, :]

        # (batch, horizon)
        x = self.output(x)

        return x

    def print_model_size(self):
        print(f'No. params: {sum(p.numel() for p in self.parameters())}')
        print(f'No. trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def save(self, path='forecast_transformer.pt', mean=None, std=None):
        ckpt = {'state_dict': self.state_dict()}
        if mean is not None:
            ckpt['mean'] = torch.as_tensor(mean)
            ckpt['std'] = torch.as_tensor(std)            
        torch.save(ckpt, path)

    def load(self, path='forecast_transformer.pt'):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state_dict'])
        self.eval()
        mean = ckpt['mean']
        std = ckpt['std']
        return (None, None) if mean is None else (mean.numpy(), std.numpy())


class ModelTrainer:

    def __init__(self, model, window_generator, batch_size, loss_fn, optimizer, epochs):
        self.model = model
        self.window_generator = window_generator
        self.random_window_generator = None
        self.batch_size = batch_size
        self._steps_per_epoch = window_generator.n_train_rows // batch_size - 1
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs

    def _get_batch_of_random_windows(self):
        batch = []
        for _ in range(self.batch_size):
            window = next(self.random_window_generator)
            batch.append(window)
        X = torch.stack([w[0] for w in batch])
        y = torch.stack([w[1] for w in batch])
        t = [w[2] for w in batch]
        return X, y, t

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.random_window_generator = self.window_generator.get_random_windows()
            epoch_loss = 0.0
            for _ in tqdm(range(self._steps_per_epoch), desc=f"Epoch {epoch + 1}/{self.epochs}"):
                X_batch, y_batch, _ = self._get_batch_of_random_windows()
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}: {epoch_loss / self._steps_per_epoch:.4f}")


class ModelEvaluator:

    def __init__(self, model, window_generator, batch_size=32, device='cpu'):
        self.model = model
        self.window_generator = window_generator
        self.batch_size = batch_size
        self.device = device
        self.times = None
        self.y_obs = None
        self.y_pred = None
        self.residuals = None

    def _batches(self):
        """Yield chronological test batches, including a final partial one."""
        gen = self.window_generator.get_chronological_windows()
        batch = []
        for window in gen:
            batch.append(window)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    @staticmethod
    def _collate(batch):
        X = torch.stack([w[0] for w in batch])
        y = torch.stack([w[1] for w in batch])
        t = np.array([w[2] for w in batch])
        return X, y, t

    def _run_inference(self):
        """Single pass: collect timestamps, observations and predictions."""
        self.model.eval()
        times_list, obs_list, pred_list = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, t_batch in self._batches():
                pred_batch = self.model(X_batch)
                times_list.append(t_batch)
                obs_list.append(y_batch.cpu().numpy())
                pred_list.append(pred_batch.cpu().numpy())

        self.times = np.concatenate(times_list)
        self.y_obs = np.concatenate(obs_list)
        self.y_pred = np.concatenate(pred_list)

        # back to original units so metrics read in degrees, not standard deviations
        wg = self.window_generator
        self.y_obs = self.y_obs * wg.target_std + wg.target_mean
        self.y_pred = self.y_pred * wg.target_std + wg.target_mean
        self.residuals = self.y_pred - self.y_obs

    def _plot_residuals_histogram(self):
        fig, ax = plt.subplots()
        ax.hist(self.residuals, bins=50)
        ax.set_xlabel('residual (pred - obs)')
        ax.set_ylabel('count')
        ax.set_title('Residuals')
        plt.tight_layout()
        plt.savefig('residual_histogram.png')
        plt.close(fig)

    def _plot_obs_pred(self, t_min=None, t_max=None, n_plot=1000, lead=0):
        times = self.times + np.timedelta64(lead, 'h')
        obs = self.y_obs[:, lead]
        pred = self.y_pred[:, lead]
        mask = np.ones_like(times, dtype=bool)
        if t_min is not None:
            mask &= times >= np.datetime64(t_min)
        if t_max is not None:
            mask &= times <= np.datetime64(t_max)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            print('No points in requested time range')
            return
        if len(idx) > n_plot:
            idx = idx[:n_plot]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(times[idx], obs[idx], c='red', s=6, label='obs')
        ax.plot(times[idx], pred[idx], '-', linewidth=1, label='pred')
        ax.set_xlabel('time')
        ax.set_ylabel(self.window_generator._columns[1])
        ax.set_title('Observed vs Predicted (test set)')
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig('y_pred_vs_y_obs.png')
        plt.close(fig)

    def print_metrics(self):
        per_step_mse = np.mean(self.residuals ** 2, axis=0)
        per_step_mae = np.mean(np.abs(self.residuals), axis=0)
        print(f'Test windows: {len(self.residuals)}')
        for h in range(self.residuals.shape[1]):
            print(f'  t+{h + 1:>3}h  RMSE {per_step_mse[h] ** 0.5:7.4f}  MAE {per_step_mae[h]:7.4f}')
        print(f'Overall RMSE: {float(np.mean(self.residuals ** 2)) ** 0.5:.4f}')

    def run(self, t_min=None, t_max=None):
        self._run_inference()
        self.print_metrics()
        self._plot_residuals_histogram()
        self._plot_obs_pred(t_min=t_min, t_max=t_max)


if __name__ == "__main__":

    SORTED_CSV = Path('data/weatherHistory_sorted.csv')
    MODEL_PATH = Path('forecast_transformer.pt')
    HORIZON = 1

    if not SORTED_CSV.exists():
        prepare()

    window_generator = WindowGenerator(
        csv_path=SORTED_CSV,
        columns=['Formatted Date', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
                 'Wind Bearing (degrees)', 'Pressure (millibars)'],
        index_column='Formatted Date',
        window_size=240,
        horizon=HORIZON,
        chunksize=5000,
        f_split=0.9,
    )

    model = ForecastTransformer(horizon=HORIZON)
    model.print_model_size()

    if MODEL_PATH.exists():
        print(f'Loading weights from {MODEL_PATH}')
        model.load(MODEL_PATH)
    else:
        model_trainer = ModelTrainer(
            model,
            window_generator,
            batch_size=32,
            loss_fn=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            epochs=5,
        )
        model_trainer.train()
        model.save(MODEL_PATH, mean=window_generator._mean, std=window_generator._std)

    model_evaluator = ModelEvaluator(model, window_generator, batch_size=32)
    model_evaluator.run()