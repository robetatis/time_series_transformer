import pytest
import numpy as np
import pandas as pd 
from time_series_transformer import WindowGenerator


class TestWindowGenerator:

    def test_build_chunk_offsets(self, tmp_path):
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2026-01-01', periods=1000, freq='h', tz='UTC'),
            'field1': np.zeros(1000),
            'field2': np.zeros(1000),
        })
        csv_path = tmp_path / 'data.csv'
        df.to_csv(csv_path, index=False, lineterminator='\n')
        
        header, chunk_offsets_train, chunk_offsets_test, nrows, n_train_rows = (
            WindowGenerator._build_chunk_offsets(
                csv_path=csv_path, 
                chunksize=100, 
                f_split=0.9,
            )
        )

        lines = csv_path.read_bytes().split(b'\n')
        for k, offset in enumerate(chunk_offsets_train + chunk_offsets_test):
            with open(csv_path, 'rb') as f:
                f.seek(offset)
                assert f.readline() == lines[1 + 100*k] + b'\n'

        assert header == ['timestamp', 'field1', 'field2']
        assert len(chunk_offsets_train) == 9
        assert len(chunk_offsets_test) == 1
        assert nrows == len(df)
        assert n_train_rows == int(0.9*len(df))


