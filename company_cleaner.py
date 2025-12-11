import csv
import requests
import time
from rapidfuzz import process, fuzz

# ----------------------------
# 1. Convert Company Name → Ticker (Yahoo Finance)
# ----------------------------
def name_to_ticker(company_name: str):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers)
        data = r.json()
        if data.get("quotes"):
            return data["quotes"][0].get("symbol")
        return None
    except:
        return None

# ----------------------------
# 2. Convert Ticker → CIK (SEC API)
# ----------------------------
def ticker_to_cik(ticker: str):
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "eveme@bu.edu"}

    try:
        r = requests.get(mapping_url, headers=headers)
        data = r.json()
        for item in data.values():
            if item["ticker"].lower() == ticker.lower():
                cik_str = str(item["cik_str"]).zfill(10)
                return cik_str
        return None
    except:
        return None

# ----------------------------
# 3. Fuzzy lookup fallback
# ----------------------------
def fuzzy_lookup(company_name, sec_mapping):
    # sec_mapping: ticker -> company_name
    best_match, score, ticker = process.extractOne(
        company_name,
        {v: k for k, v in sec_mapping.items()},
        scorer=fuzz.token_sort_ratio
    )
    if score >= 80:  # threshold for matching
        return ticker
    return None

# ----------------------------
# 4. Main pipeline
# ----------------------------
def process_names(input_csv):
    # Load company names
    names = []
    with open(input_csv, "r", encoding="utf-8") as fin:
        for line in fin:
            name = line.strip().strip('"')
            if name:
                names.append(name)

    # Load SEC mapping for fuzzy lookup
    sec_mapping_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "eveme@bu.edu"}
    sec_mapping = {}
    try:
        r = requests.get(sec_mapping_url, headers=headers)
        data = r.json()
        for item in data.values():
            sec_mapping[item["ticker"]] = item["title"]
    except:
        pass

    results = []
    cik_list = []
    missing_companies = []

    for name in names:
        print(f"Looking up ticker for: {name}...")
        ticker = name_to_ticker(name)
        time.sleep(0.3)

        if ticker is None:
            # Try fuzzy lookup
            ticker = fuzzy_lookup(name, sec_mapping)
            if ticker:
                print(f"  -> Recovered by fuzzy match: {ticker}")
            else:
                print("  -> Not found, skipping.")
                missing_companies.append(name)
                continue
        else:
            print(f"  -> Ticker: {ticker}")

        print(f"Looking up CIK for: {ticker}...")
        cik = ticker_to_cik(ticker)
        time.sleep(0.3)

        if cik is None:
            print("  -> CIK not found, skipping.")
            missing_companies.append(name)
            continue

        print(f"  -> CIK: {cik}")
        results.append([name, ticker, cik])
        cik_list.append([cik])

    # Write filtered name+ticker+CIK
    with open("tickers_clean.csv", "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["company_name", "ticker", "cik"])
        writer.writerows(results)

    # Write CIK-only file (for 10-K downloader)
    with open("cik_list.csv", "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for cik in cik_list:
            writer.writerow(cik)

    # Write companies not found
    with open("not_found.csv", "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["company_name"])
        for name in missing_companies:
            writer.writerow([name])

    print("\n🎉 Done! Saved:")
    print("  - tickers_clean.csv  (company, ticker, cik)")
    print("  - cik_list.csv       (just CIKs for downloading)")
    print("  - not_found.csv      (companies not found)")

if __name__ == "__main__":
    process_names("company_ticker_list.csv")
