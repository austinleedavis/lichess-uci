"""
Prepares a single month of games from the the Lichess Open Database
"""

import argparse
import datetime
import glob
import os
import multiprocessing
import subprocess
from io import TextIOWrapper
import json

import datasets
import regex as re
import requests
import zstandard as zstd
from huggingface_hub import whoami
from tqdm.auto import tqdm

COUNTS_URL = "https://database.lichess.org/standard/counts.txt"
HEADERS = [
    "Event",
    "Site",
    "White",
    "Black",
    "Result",
    "UTCDate",
    "UTCTime",
    "WhiteElo",
    "BlackElo",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "ECO",
    "Opening",
    "TimeControl",
    "Termination",
    "Transcript",
]

UCI_PATTERN = re.compile(r"\b[a-h][1-8][a-h][1-8][QBRN]?\b")
FEN_PATTERN = re.compile(r"\{ ([^}]+) \}")


def parse_moves_and_fen(game_string: str):
    uci_moves = UCI_PATTERN.findall(game_string)
    fen_strings = FEN_PATTERN.findall(game_string)
    return " ".join(uci_moves), json.dumps(fen_strings)


def parse_moves_only(game_string: str):
    uci_moves = UCI_PATTERN.findall(game_string)
    return " ".join(uci_moves)


def main():
    ############################
    # Setup Parser
    ############################
    ZST_RECORDS = parse_counts_file(COUNTS_URL)

    parser = argparse.ArgumentParser(
        description="Utility to process Lichess.org open dataset files into Huggingface datsets in UCI format."
    )
    # fmt: off
    parser.add_argument("--year", choices=sorted(list(set([k[0] for k in ZST_RECORDS.keys()]))), help="Year (format: yyyy). Dynamically updated from Lichess.org", required=False)    
    parser.add_argument("--month", choices=sorted(list(set([k[1] for k in ZST_RECORDS.keys()]))), help="Month (format: mm). Dynamically updated from Lichess.org", required=False)    
    parser.add_argument("--push_to_hub", action="store_true", help="Push dataset to huggingface hub.")
    parser.add_argument("--data_dir", type=str,default="data",help="Location to store the data. (Default=data/)")
    parser.add_argument("--list",action="store_true", help="Print the list of available files from the Lichess.org Open Dataset",)    
    parser.add_argument("--missing_only",action="store_true",help="When listing, only show files which are not currently in the download cache.",)
    parser.add_argument("--force_download_zst",action="store_true",help="Download (overwrite) existing ZST file",)
    parser.add_argument("--force_overwrite_pgn",action="store_true",help="Decompress (overwrite) existing PGN file",)
    parser.add_argument("--force_overwrite_uci",action="store_true",help="Recreate (overwrite) existing UCI file by processing the raw PGN again.",)
    parser.add_argument("--force_overwrite_tsv",action="store_true",help="Recreate (overwrite) existing TSV file by processing the UCI file again.",)
    parser.add_argument("--include_fen",action="store_true",help="Include FEN strings in the TSV file.",)    
    # fmt: on

    args = parser.parse_args()

    # ensure forcing upstream changes forces dependencies to be re-computed, too
    if args.force_download_zst:
        args.force_overwrite_pgn = True
    if args.force_overwrite_pgn:
        args.force_overwrite_uci = True
    if args.force_overwrite_uci:
        args.force_overwrite_tsv = True

    ############################
    # List
    ############################
    record_list = list(ZST_RECORDS.values())

    if args.missing_only:
        record_list = get_missing_cache_records(record_list)

    if args.list:
        for i, record in enumerate(sorted(record_list, key=lambda r: r["date"])):
            print(f"{i}:\tYYYYmm: '{record['date']}', count: {int(record['count']):,d}")
        exit(0)

    ############################
    # prepare paths
    ############################
    assert (
        args.year,
        args.month,
    ) in ZST_RECORDS.keys(), f"The games for {args.month}/{args.year} are not available from the Lichess Open Database."

    selected_record = ZST_RECORDS[(args.year, args.month)]
    data_folder = os.path.join(args.data_dir, selected_record["date"])
    os.makedirs(data_folder, exist_ok=True)

    pgn_path = os.path.join(data_folder, selected_record["pgn"])
    uci_path = os.path.join(data_folder, selected_record["uci"])
    tsv_path = os.path.join(data_folder, selected_record["tsv"])
    log_path = os.path.join(data_folder, selected_record["log"])

    print(f"Processing {args.year}{args.month}...")

    ############################
    # Download and extract
    ############################
    if args.force_download_zst or (not is_cached(selected_record["url"])):
        download(url=selected_record["url"], force=args.force_download_zst)

    if args.force_overwrite_pgn or (not os.path.exists(pgn_path)):
        cache_path = get_cache_path(selected_record["url"])
        extract_zst_file(cache_path, pgn_path)

    ############################
    # PGN to UCI
    ############################
    if args.force_overwrite_uci or (not os.path.exists(uci_path)):

        # Create a temporary headers file so we can use pgn-extract
        headers_file = os.path.join(data_folder, "lichess_headers.temp.txt")
        with open(headers_file, "w") as f:
            f.writelines(
                "\n".join(HEADERS[:-1])
            )  # skip final header: it's not in the PGN file!!!

        # Convert PGN to UCI
        print(f"Running pgn-extract. Logging to: {log_path}")
        print(f"Total Games to process: {int(selected_record['count']):,d}")
        # the process call outputs uci moves (with uppercase promotions), skips faux en passant, and uses 100k char length to avoid multi-line transcripts
        process_call = f"pgn-extract -L{log_path} -R{headers_file} -Wuci --nofauxep -w100000 {pgn_path} -o {uci_path}"
        if args.include_fen:
            process_call += " --fencomments"
        subprocess.run(process_call.split(" "))
        os.remove(headers_file)  # remove temp file

    ############################
    # UCI to TSV
    ############################
    if args.force_overwrite_tsv or (not os.path.exists(tsv_path)):

        # We append the header here because pgn-extract doesn't recognize it as a valid header.
        if args.include_fen:
            HEADERS.append("Fens")

        # do a quick line count so we can show progess w/ tqdm
        uci_file_line_count = count_lines(uci_path)
        with open(uci_path, "r") as in_file, open(tsv_path, "w") as out_file:

            # write the TSV headers
            write_line(out_file=out_file, headers=HEADERS, data=None)

            data = {}
            for line in tqdm(
                in_file, total=uci_file_line_count, desc="Converting to TSV"
            ):
                if len(line) < 3:  # blanks
                    continue

                if line[0] != "[":  # uci line
                    if args.include_fen:
                        data["Transcript"], data["Fens"] = parse_moves_and_fen(line)
                    else:
                        data["Transcript"] = parse_moves_only(line)
                    write_line(out_file=out_file, headers=HEADERS, data=data)
                    data = {}
                    continue

                match = re.match(r'\[(\w+) "([^"]+)"\]', line)
                if not match:  # line doesn't contain one of our HEADERS
                    continue

                # process the header
                key, value = match.groups()

                if key not in HEADERS:
                    continue

                if key == "Site":
                    # only include the UUID, not the whole URL
                    data[key] = value.split("/")[-1]
                else:
                    data[key] = value

    ############################
    # Push to Hub
    ############################
    if args.push_to_hub:
        ds = datasets.Dataset.from_csv(tsv_path, delimiter="\t")
        ds = fix_dtypes(ds, has_fens=args.include_fen)
        ds_name = f"{whoami()['name']}/lichess-uci{'-fens' if args.include_fen else ''}"
        # save to disk (in case an issue arises in upload)
        local_output_file = os.path.join(data_folder, ds_name)
        os.makedirs(local_output_file, exist_ok=True)
        ds.save_to_disk(local_output_file)

        # push to hub
        ds.push_to_hub(
            repo_id=ds_name,
            config_name=f"{selected_record['date']}",
            split="train",
            data_dir=f"data/{selected_record['date']}",
        )
    print("Finished.")


def get_cache_path(url: str):
    config = datasets.DownloadConfig(local_files_only=True)
    manager = datasets.DownloadManager(download_config=config)
    cache_path = manager.download_and_extract(url)
    return cache_path


def count_lines(path: str) -> str:
    """Counts the number of lines in a file."""

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line_count = sum(bl.count("\n") for bl in blocks(f))
    return line_count


def write_line(*, out_file: TextIOWrapper, headers: list[str], data: dict[str, str]):
    if data:
        element_list = [data.get(header, "") for header in headers]
        out_file.write("\t".join([element for element in element_list]) + "\n")
    else:
        out_file.write("\t".join(headers) + "\n")


def parse_array(example):
    example["Fens"] = json.loads(example["Fens"])
    return example


def fix_dtypes(ds: datasets.Dataset, has_fens: bool = False):
    num_proc = min(8, multiprocessing.cpu_count() // 2)

    if has_fens:
        ds = ds.map(parse_array, num_proc=num_proc, desc="Parsing FEN arrays")

    ds = ds.map(
        map_dtypes,
        input_columns=["WhiteElo", "BlackElo", "UTCDate", "UTCTime"],
        desc="Fixing dtypes",
        num_proc=num_proc,
    )

    ds.features["WhiteElo"] = datasets.Value("int32")
    ds.features["BlackElo"] = datasets.Value("int32")
    ds.features["UTCDate"] = datasets.Value("date32")
    ds.features["UTCTime"] = datasets.Value("time32[s]")

    return ds


def map_dtypes(whiteelo: str, blackelo: str, utcdate: str, utctime: str):

    whiteelo = int(whiteelo) if type(whiteelo) is str and whiteelo.isdecimal() else 0
    blackelo = int(blackelo) if type(blackelo) is str and blackelo.isdecimal() else 0

    if whiteelo == 0 and blackelo > 0:
        whiteelo = blackelo
    elif whiteelo > 0 and blackelo == 0:
        blackelo = whiteelo

    utcdate = datetime.date(*[int(x) for x in utcdate.split(".")])
    utctime = datetime.time(*[int(x) for x in utctime.split(":")])

    return {
        "WhiteElo": whiteelo,
        "BlackElo": blackelo,
        "UTCDate": utcdate,
        "UTCTime": utctime,
    }


def is_cached(zst_filename: str) -> bool:
    config = datasets.DownloadConfig(local_files_only=True)
    manager = datasets.DownloadManager(download_config=config)
    try:
        manager.download(zst_filename)
    except FileNotFoundError as e:
        cache_path = re.search(r"(\/[\w\d\.]+)+", str(e)).group(0)
        cache_dir = os.path.dirname(cache_path)
        cache_filename = os.path.basename(cache_path)
        file_list = glob.glob(pathname=f"{cache_filename}*", root_dir=cache_dir)
        file_list = [f for f in file_list if f[-5] != "."]
        if not file_list:
            return False
    return True


def download(url, force):
    print(f"Downloading: {url}")
    config = datasets.DownloadConfig(force_download=force)
    manager = datasets.DownloadManager(download_config=config)
    cache_path = manager.download_and_extract(url)
    print("cache_path: ", cache_path)
    return cache_path


def extract_zst_file(cache_path, out_file):
    print(f"Extracting to: {out_file}")
    dctx = zstd.ZstdDecompressor()
    with open(cache_path, "rb") as ifh, open(out_file, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)


def get_missing_cache_records(zst_records_list: list[dict]) -> dict:
    """
    Returns the subset of the Lichess.org Open Dataset files that are not already cached on the machine.
    """
    missing_records = []
    for record in zst_records_list:
        zst_file_url = record["url"]
        if not is_cached(zst_file_url):
            missing_records.append(record)

    return missing_records


def parse_counts_file(url: str):
    """
    Parses the official list of"""
    response = requests.get(url)
    response.raise_for_status()  # Ensures we handle any HTTP errors
    lines = response.text.splitlines()  # Splits text content by lines
    lines = [line.split(" ") for line in lines if len(line)]
    records = {}
    for line in lines:
        yyyy = line[0][26:30]
        mm = line[0][31:33]
        records[(yyyy, mm)] = dict(
            filename=line[0],
            url=f"https://database.lichess.org/standard/{line[0]}",
            count=line[1],
            date=f"{yyyy}{mm}",
            pgn=line[0][:-4],
            uci=line[0][:-7] + "uci",
            tsv=line[0][:-7] + "tsv",
            log=line[0][:-7] + "log",
        )

    return records


if __name__ == "__main__":
    main()
