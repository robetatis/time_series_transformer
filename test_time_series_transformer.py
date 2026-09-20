import pytest
import numpy as np
import pandas as pd 
from time_series_transformer import WindowGenerator

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
            'timestamp': pd.date_range(start='2026-01-01', periods=1000, freq='h', tz='UTC'),
            'field1': np.arange(1000),
            'field2': np.arange(1000)*2,
    })

class TestWindowGenerator:

    def test_build_chunk_offsets(self, tmp_path, dummy_df):
        csv_path = tmp_path / 'data.csv'
        dummy_df.to_csv(csv_path, index=False, lineterminator='\n')
        CHUNKSIZE = 100
        F_SPLIT = 0.9
        header, chunk_offsets_train, chunk_offsets_test, nrows, n_train_rows = (
            WindowGenerator._build_chunk_offsets(
                csv_path=csv_path, 
                chunksize=CHUNKSIZE, 
                f_split=F_SPLIT,
            )
        )

        lines = csv_path.read_bytes().split(b'\n')
        for k, offset in enumerate(chunk_offsets_train + chunk_offsets_test):
            with open(csv_path, 'rb') as f:
                f.seek(offset)
                assert f.readline() == lines[1 + CHUNKSIZE*k] + b'\n'

        assert header == ['timestamp', 'field1', 'field2']
        assert len(chunk_offsets_train) == 9
        assert len(chunk_offsets_test) == 1
        assert nrows == len(dummy_df)
        assert n_train_rows == int(0.9*len(dummy_df))

    def test_read_chunk(self, tmp_path, dummy_df):

        csv_path = tmp_path / 'data.csv'
        dummy_df.to_csv(csv_path, index=False, lineterminator='\n')

        CHUNKSIZE = 100

        # build a byte offset by reading the file and splitting lines
        # byte offset is the size of all content before the desired
        # position, here 2 x CHUNKSIZE
        lines = csv_path.read_bytes().splitlines(keepends=True)

        offset_mid_file = sum(len(line) for line in lines[:(1 + 2*CHUNKSIZE)])
        offset_end_of_file = sum(len(line) for line in lines[:(1 + int(9.5*CHUNKSIZE))])

        arr_mid_file, idx_mid_file = WindowGenerator._read_chunk(
            csv_path, 
            offset_mid_file,
            ['timestamp', 'field1', 'field2'], 
            ['timestamp', 'field1', 'field2'],  
            'timestamp',
            CHUNKSIZE
        )

        arr_end_of_file, idx_end_of_file = WindowGenerator._read_chunk(
            csv_path, 
            offset_end_of_file,
            ['timestamp', 'field1', 'field2'], 
            ['timestamp', 'field1', 'field2'],  
            'timestamp',
            CHUNKSIZE
        )

        expected_arr_mid_file = dummy_df[['field1', 'field2']].iloc[(CHUNKSIZE*2):(CHUNKSIZE*2 + CHUNKSIZE)].to_numpy()
        expected_idx_mid_file = dummy_df['timestamp'].iloc[(CHUNKSIZE*2):(CHUNKSIZE*2 + CHUNKSIZE)].values

        expected_arr_end_of_file = dummy_df[['field1', 'field2']].iloc[(int(CHUNKSIZE*9.5)):].to_numpy()
        expected_idx_end_of_file = dummy_df['timestamp'].iloc[(int(CHUNKSIZE*9.5)):].values

        assert arr_mid_file.shape == (CHUNKSIZE, 2) # two cols: 'field1' and 'field2'
        assert len(idx_mid_file) == CHUNKSIZE
        np.testing.assert_array_equal(arr_mid_file, expected_arr_mid_file)
        np.testing.assert_array_equal(idx_mid_file, expected_idx_mid_file)

        assert arr_end_of_file.shape == (50, 2) # two cols: 'field1' and 'field2'
        assert len(idx_end_of_file) == 50
        np.testing.assert_array_equal(arr_end_of_file, expected_arr_end_of_file)
        np.testing.assert_array_equal(idx_end_of_file, expected_idx_end_of_file)


