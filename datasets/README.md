# datasets

Sample market data used for training and evaluating agents. Each CSV file corresponds to a currency pair or instrument (e.g. `EURUSD.csv`, `XAUUSD.csv`).

## Required format

Files must contain at least the following columns:

- `time` – timestamp of the bar
- `open`
- `high`
- `low`
- `close`
- `tick_volume`

Additional columns (such as `spread` or `real_volume`) are ignored. Files are sorted by `time` when loaded.

The utilities in `drl_modules/data_extract.py` automatically reformat files that lack the required headers.

To use your own data, place the CSV in this directory and select it when running `main.py`.
