# Lichess Open Database UCI Converter

The [Lichess.org Open Database](https://database.lichess.org/) (LOD) publishes standard rated games, played on lichess.org, in PGN format. This repository provides scripts to download and decompress those files, convert the moves from Portable Game Notation (PGN) format to Universal Chess Interface (UCI) format, and load the games into a Hugging Face dataset. This allows for efficient storage and easy access to game data in a format widely compatible with machine learning models and analyses.

## Features

- Converts large `.zst` files from PGN to UCI.
- Integrates with `pgn-extract` for precise extraction and formatting of PGN files.
- Outputs a Hugging Face dataset compatible with popular frameworks for machine learning and data processing.

## Requirements

- [`pgn-extract`](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
- Additional Python packages as specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/austinleedavis/lichess-uci.git
   cd lichess-uci
   ```


1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

1. Install `pgn-extract` following the instructions on its [official site](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/).

## Usage

```bash
prepare.py [-h] [--list] [--year {2013,...}] [--month {01, ..., 12}]
                  [--missing_only] [--push_to_hub] [--data_dir DATA_DIR] [--force_download_zst] [--force_overwrite_pgn] [--force_overwrite_uci]
                  [--force_overwrite_tsv] [--download_proc DOWNLOAD_PROC] [--process_proc PROCESS_PROC]
```
### Options
```text
-h, --help             Show this help message and exit
--year {2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024}
                       Year (format: yyyy). Dynamically updated from Lichess.org
--month {01,02,03,04,05,06,07,08,09,10,11,12}
                       Month (format: mm). Dynamically updated from Lichess.org
--push_to_hub          Push dataset to huggingface hub.
--data_dir DATA_DIR    Location to store the processed data. . (Default=data/)
--list                 Print the list of available files from the Lichess.org Open Dataset
--missing_only         When listing available files, only show files which are not currently in the download cache.
--force_download_zst   Download (overwrite) existing ZST file, i.e., ignore the download cache
--force_overwrite_pgn  Decompress (overwrite) existing PGN file
--force_overwrite_uci  Recreate (overwrite) existing UCI file by processing the raw PGN again.
--force_overwrite_tsv  Recreate (overwrite) existing TSV file by processing the UCI file again.
```

## Output

The output will be a Hugging Face dataset in UCI format that can be directly loaded for analysis, training, or other processing needs.

## Examples

Here’s an example of converting a file and pushing the resulting dataset to your HuggingFace Hub account:

```bash
python prepare.py --year 2013 --month 01 --push_to_hub
```
The converted dataset can be loaded in two ways:

1. From the local *.tsv file:

   ```python
   from datasets import Dataset

   ds = Dataset.from_csv("data/201301/*.tsv", sep='\t')
   print(ds)
   ```
1. From the Huggingface Hub:
   ```python 
   from datasets import load_dataset
   ds = load_dataset('<your_username>/lichess-uci', '201301', split='train')
   print(ds)
   ```


## License

This project is licensed under the MIT License. See the `LICENSE` file for more information. Lichess games and puzzles are released under the Creative Commons CC0 license.

## Acknowledgements

- [Lichess.org Open Dataset](https://database.lichess.org/) for the game data.
- [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) for PGN extraction and conversion tools.
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) for dataset handling.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
