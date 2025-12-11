"""
A standalone script to extract MD&A (Management's Discussion and Analysis) sections
from downloaded 10-K forms. The script processes all 10-K files in a directory and
extracts the ITEM 7 section, saving them as .mda files.
"""

import argparse
import concurrent.futures
import os
import re
import time
import unicodedata
from glob import glob


# -----------------------------
# Utility functions
# -----------------------------
def timeit(func):
    """Decorator to time functions"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper


# -----------------------------
# MD&A Extraction functions
# -----------------------------
def normalize_text(text):
    """Normalize Text"""
    # Remove HTML tags if present (simple regex-based removal)
    # This handles cases where files contain HTML markup
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    
    # Replace common HTML entities (including numeric entities)
    text = re.sub(r'&nbsp;|&#160;|&#xa0;', ' ', text)  # Non-breaking space
    text = re.sub(r'&amp;|&#38;', '&', text)
    text = re.sub(r'&lt;|&#60;', '<', text)
    text = re.sub(r'&gt;|&#62;', '>', text)
    text = re.sub(r'&rsquo;|&#8217;|&#x2019;', "'", text)  # Right single quote
    text = re.sub(r'&lsquo;|&#8216;|&#x2018;', "'", text)  # Left single quote
    text = re.sub(r'&mdash;|&#8212;|&#x2014;', '—', text)  # Em dash
    text = re.sub(r'&ndash;|&#8211;|&#x2013;', '–', text)  # En dash
    # Replace other numeric entities
    text = re.sub(r'&#\d+;', ' ', text)  # Other numeric entities
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)  # Hex entities
    
    text = unicodedata.normalize("NFKD", text)  # Normalize
    text = "\n".join(text.splitlines())  # Unicode break lines

    # Convert to upper
    text = text.upper()  # Convert to upper

    # Take care of breaklines & whitespaces combinations
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)  # Multiple spaces to single space

    # To find MDA section, reformat item headers
    text = text.replace("\n.\n", ".\n")  # Move Period to beginning

    text = text.replace("\nI\nTEM", "\nITEM")
    text = text.replace("\nITEM\n", "\nITEM ")
    text = text.replace("\nITEM  ", "\nITEM ")
    text = text.replace("ITEM  ", "ITEM ")  # Handle multiple spaces

    text = text.replace(":\n", ".\n")

    # Math symbols for clearer looks
    text = text.replace("$\n", "$")
    text = text.replace("\n%", "%")

    # Reformat
    text = text.replace("\n", "\n\n")  # Reformat by additional breakline

    return text


def write_content(content, file_path):
    """Write content to file"""
    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:  # Only create if there's a directory path
        os.makedirs(output_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fout:
        fout.write(content)


def find_mda_from_text(text, start=0):
    """Find MDA section from normalized text
    
    Args:
        text (str): Normalized text to search
        start (int): Starting position in text to begin search
    
    Returns:
        tuple: (mda_text, end_position)
    """
    debug = False

    mda = ""
    end = 0
    begin = -1

    # Define start & end signal for parsing
    # More flexible patterns to catch variations
    item7_begins = [
        "\nITEM 7.",
        "\nITEM 7 –",
        "\nITEM 7:",
        "\nITEM 7 ",
        "\nITEM 7\n",
        "ITEM 7.",
        "ITEM 7 –",
        "ITEM 7:",
        "ITEM 7 ",
        "ITEM 7\n",
        "\n\nITEM 7.",
        "\n\nITEM 7 ",
        "ITEM 7.",  # At start of text
    ]
    item7_ends = ["\nITEM 7A", "\n\nITEM 7A", "ITEM 7A", "\nITEM 7A.", "\nITEM 7A ", "ITEM 7A."]
    if start != 0:
        item7_ends.extend(["\nITEM 7", "\n\nITEM 7", "ITEM 7"])  # Case: ITEM 7A does not exist
    item8_begins = ["\nITEM 8", "\n\nITEM 8", "ITEM 8", "\nITEM 8.", "\nITEM 8 ", "ITEM 8."]

    """
    Parsing code section
    """
    text = text[start:]

    # Get begin - look for ITEM 7 with "MANAGEMENT" or "DISCUSSION" nearby
    # This helps identify the correct ITEM 7 (MD&A) vs other ITEM 7s
    for item7 in item7_begins:
        begin = text.find(item7)
        if begin != -1:
            # Check if this ITEM 7 is followed by MD&A keywords (within next 300 chars)
            # This helps distinguish MD&A from other Item 7 sections
            check_text = text[begin:begin+300]
            has_mda_keywords = (
                "MANAGEMENT" in check_text or 
                "DISCUSSION" in check_text or 
                "ANALYSIS" in check_text
            )
            
            # Also check if it's NOT a different Item 7 (like "Item 7. Financial Statements")
            is_not_financial_statements = "FINANCIAL STATEMENTS" not in check_text[:100]
            
            if has_mda_keywords and is_not_financial_statements:
                if debug:
                    print(f"Found ITEM 7 (MD&A) at {begin}: {item7}")
                break
            # If not MD&A, continue searching
            begin = -1

    if begin == -1:
        # Fallback: try to find any ITEM 7 and check broader context
        for item7 in item7_begins:
            begin = text.find(item7)
            if begin != -1:
                # Check broader context (500 chars) for MD&A keywords
                check_text = text[begin:begin+500]
                if "MANAGEMENT" in check_text or "DISCUSSION" in check_text:
                    if debug:
                        print(f"Found ITEM 7 (fallback) at {begin}: {item7}")
                    break
                begin = -1

    if begin != -1:  # Begin found
        for item7A in item7_ends:
            end = text.find(item7A, begin + 1)
            if debug:
                print(f"Checking end pattern {item7A}: {end}")
            if end != -1:
                break

        if end == -1:  # ITEM 7A does not exist
            for item8 in item8_begins:
                end = text.find(item8, begin + 1)
                if debug:
                    print(f"Checking ITEM 8 pattern {item8}: {end}")
                if end != -1:
                    break

        # Get MDA
        if end > begin:
            mda = text[begin:end].strip()
        else:
            end = 0

    return mda, end


def parse_mda(form_path, mda_path, overwrite=False):
    """Reads form and parses mda
    
    Args:
        form_path (str): Path to the 10-K form file
        mda_path (str): Path where MD&A section will be saved
        overwrite (bool): Whether to overwrite existing MD&A files
    """
    if not overwrite and os.path.exists(mda_path):
        print("{} already exists.  Skipping parse mda...".format(mda_path))
        return
    
    # Read
    print("Parse MDA {}".format(form_path))
    try:
        with open(form_path, "r", encoding="utf-8", errors="ignore") as fin:
            text = fin.read()
    except Exception as e:
        print("Error reading {}: {}".format(form_path, e))
        return

    # Check if file is empty or too small
    if len(text) < 100:
        print("Parse MDA failed {}: File too small or empty".format(form_path))
        return

    # Normalize text here
    text = normalize_text(text)

    # Parse MDA
    mda, end = find_mda_from_text(text)
    # Parse second time if first parse results in index
    if mda and len(mda.encode("utf-8")) < 1000:
        mda, _ = find_mda_from_text(text, start=end)

    if mda and len(mda.strip()) > 100:  # Ensure we have substantial content
        print("Write MDA to {} ({} bytes)".format(mda_path, len(mda.encode("utf-8"))))
        write_content(mda, mda_path)
    else:
        print("Parse MDA failed {}: No MD&A section found or content too short".format(form_path))


@timeit
def parse_mda_multiprocess(form_dir: str, mda_dir: str, overwrite: bool = False):
    """Parse MDA section from forms with multiprocess
    
    Args:
        form_dir (str): Directory containing downloaded 10-K forms
        mda_dir (str): Directory where MD&A sections will be saved
        overwrite (bool): Whether to overwrite existing MD&A files
    """
    # Create output directory
    os.makedirs(mda_dir, exist_ok=True)

    # Prepare arguments
    form_paths = sorted(glob(os.path.join(form_dir, "*")))
    
    if not form_paths:
        print(f"No files found in {form_dir}")
        return
    
    # Filter out directories and build corresponding mda_paths
    actual_form_paths = []
    actual_mda_paths = []
    for form_path in form_paths:
        # Skip if it's a directory
        if os.path.isdir(form_path):
            continue
        form_name = os.path.basename(form_path)
        root, _ = os.path.splitext(form_name)
        mda_path = os.path.join(mda_dir, "{}.mda".format(root))
        actual_form_paths.append(form_path)
        actual_mda_paths.append(mda_path)
    
    if not actual_form_paths:
        print(f"No files to process in {form_dir}")
        return
    
    print(f"Found {len(actual_form_paths)} forms to process")
    print(f"Output directory: {os.path.abspath(mda_dir)}")

    # Multiprocess
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(parse_mda, form_path, mda_path, overwrite) 
                   for form_path, mda_path in zip(actual_form_paths, actual_mda_paths)]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
                # Note: We can't easily track success/failure from here since parse_mda doesn't return a value
                # But the print statements will show what happened
            except Exception as e:
                print(f"Error processing file: {e}")
    
    print(f"\nProcessing complete. Check output directory: {os.path.abspath(mda_dir)}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract MD&A sections from downloaded 10-K forms"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Directory containing downloaded 10-K form files"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None,
        help="Directory where extracted MD&A sections will be saved (default: ./data/mda_extracted)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing MD&A files if they exist"
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return

    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory")
        return

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join("data", "mda_extracted")
        print(f"Output directory not specified. Using default: {os.path.abspath(args.output_dir)}")

    # Extract MD&A sections
    parse_mda_multiprocess(args.input_dir, args.output_dir, args.overwrite)
    print("MD&A extraction completed!")


if __name__ == "__main__":
    main()

