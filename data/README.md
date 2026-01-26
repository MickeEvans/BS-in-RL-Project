# Data Directory

This directory is for storing data files. It is **not tracked by git** (see `.gitignore`).

## Structure

- `raw/`: Raw data files (market data, historical prices, etc.)
- `processed/`: Processed/cleaned data ready for use

## Usage

Place your data files here:
```
data/
├── raw/
│   ├── market_data.csv
│   └── option_prices.csv
└── processed/
    ├── train_data.pkl
    └── test_data.pkl
```

## Important Notes

- **Do not commit large data files** to git
- Use `.gitkeep` files to track empty directories
- Document your data sources in a `DATA_SOURCES.md` file
- Consider using cloud storage or data versioning tools for large datasets

## Data Sources

Document your data sources here or in a separate `DATA_SOURCES.md` file:
- Source name
- Download link
- Date accessed
- License information
