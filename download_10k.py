"""
A standalone script to download 10-K forms from SEC EDGAR for selected companies
based on a CSV file containing tickers and CIKs.
All downloaded forms will be saved under a folder called '10k_filings'.
"""

import argparse
import csv
import concurrent.futures
import itertools
import os
import time
from glob import glob

import requests
from ratelimit import limits, sleep_and_retry

# -----------------------------
# User configuration
# -----------------------------
headers = {
    "User-Agent": "eveme@bu.edu",  # Must be a valid email
    "Accept-Encoding": "gzip",
    "Host": "www.sec.gov",
}

SEC_GOV_URL = "https://www.sec.gov/Archives"
FORM_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{}/QTR{}/form.idx"

INDEX_HEADERS = ["Form Type", "Company Name", "CIK", "Date Filed", "File Name", "Url"]

CALLS = 10
RATE_LIMIT = 1

# -----------------------------
# Utility functions
# -----------------------------
@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def download_file(url: str, download_path: str, overwrite: bool = False):
    """Download file to disk"""
    if not overwrite and os.path.exists(download_path):
        print(f"{download_path} already exists. Skipping...")
        return True
    try:
        print(f"Requesting {url}")
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        with open(download_path, "w", encoding="utf-8") as fout:
            fout.write(res.text)
        print(f"Saved to {download_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def timeit(func):
    """Decorator to time functions"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper


# -----------------------------
# Core functions
# -----------------------------
@timeit
def download_indices(years, quarters, index_dir, overwrite=False):
    """Download EDGAR form indices"""
    os.makedirs(index_dir, exist_ok=True)
    urls = [FORM_INDEX_URL.format(year, qtr) for year, qtr in itertools.product(years, quarters)]
    download_paths = [os.path.join(index_dir, f"year{year}.qtr{qtr}.idx") 
                      for year, qtr in itertools.product(years, quarters)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, path, overwrite) for url, path in zip(urls, download_paths)]
        for f in concurrent.futures.as_completed(futures):
            f.result()


def parse_line_to_record(line, fields_begin):
    """Parse a line in form.idx"""
    record = []
    fields_indices = fields_begin + [len(line)]
    for begin, end in zip(fields_indices[:-1], fields_indices[1:]):
        field = line[begin:end].rstrip().strip('"')
        record.append(field)
    return record


@timeit
def combine_indices_to_csv(index_dir, cik_set=None):
    """
    Combine all index files into one CSV.
    cik_set: a set of CIKs (zero-padded 10-digit strings) to filter for download
    """
    rows = []

    for index_path in sorted(glob(os.path.join(index_dir, "*.idx"))):
        with open(index_path, "r", encoding="utf-8", errors="ignore") as fin:
            lines = fin.readlines()

        # Skip header lines until the dashed separator
        data_started = False
        for line in lines:
            if line.startswith("-----"):
                data_started = True
                continue
            if not data_started:
                continue

            # Each line: Form Type | Company Name | CIK | Date Filed | File Name
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # malformed line

            form_type = parts[0]
            cik = parts[-3].zfill(10)  # ensure 10-digit CIK
            date_filed = parts[-2]
            filename = parts[-1]
            company_name = " ".join(parts[1:-3])

            if form_type not in ["10-K", "10-K/A"]:
                continue
            if cik_set and cik not in cik_set:
                continue

            url = f"{SEC_GOV_URL}/{filename}".replace("\\", "/")
            rows.append([form_type, company_name, cik, date_filed, filename, url])

    # Save combined CSV
    csv_file = os.path.join(index_dir, "combined.csv")
    os.makedirs(index_dir, exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(INDEX_HEADERS)
        writer.writerows(rows)

    print(f"Combined CSV saved to {csv_file}")
    print(f"Total 10-K forms found: {len(rows)}")



def read_url_from_combined_csv(csv_path):
    urls = []
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        next(reader)  # skip header
        for row in reader:
            urls.append(row[-1])
    return urls


@timeit
def download_forms(index_dir, form_dir, overwrite=False, debug=False):
    """Download 10-K forms from combined CSV"""
    os.makedirs(form_dir, exist_ok=True)
    combined_csv = os.path.join(index_dir, "combined.csv")
    urls = read_url_from_combined_csv(combined_csv)

    download_paths = [os.path.join(form_dir, "_".join(url.split("/")[-2:])) for url in urls]

    if debug:
        download_paths = download_paths[:10]
        urls = urls[:10]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, path, overwrite) for url, path in zip(urls, download_paths)]
        for f in concurrent.futures.as_completed(futures):
            f.result()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--years", type=int, nargs="+", required=True,
        help="List of years to download (e.g., 2000 2008 2020)"
    )
    parser.add_argument("-q", "--quarters", type=int, nargs="+", default=[1,2,3,4])
    parser.add_argument("-d", "--data_dir", type=str, default="./data")
    parser.add_argument("--company_cik_list", type=str, required=True,
                        help="CSV file containing ticker and CIK (two columns: ticker,CIK)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # -----------------------------
    # Load CIK list from CSV
    # -----------------------------
    filter_set = set()
    with open(args.company_cik_list, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if row and row[0]:
                filter_set.add(row[0].strip())  # use CIK for filtering

    index_dir = os.path.join(args.data_dir, "index")
    form_dir = os.path.join(args.data_dir, "10k_filings")

    download_indices(args.years, args.quarters, index_dir, args.overwrite)
    combine_indices_to_csv(index_dir, filter_set)
    download_forms(index_dir, form_dir, args.overwrite, args.debug)


if __name__ == "__main__":
    main()
