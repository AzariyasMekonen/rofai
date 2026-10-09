"""Download magazines, newsletters, and reports from selamchildrenvillage.org.

Saves PDFs to data/sources/scraped/ and verifies text extraction.
"""

import sys
import urllib.request
from pathlib import Path
import pypdf

ROOT_DIR = Path(__file__).resolve().parents[2]
DEST_DIR = ROOT_DIR / "data" / "sources" / "scraped"

PUBLICATIONS = [
    # Anniversary Magazines
    (
        "selam_30th_anniversary_magazine.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/08/30TH-YEAR-SELAM-MAGAZIN-FINAL-1.pdf",
    ),
    (
        "selam_35th_anniversary_magazine.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/10/Selam-35th-Magazine.pdf",
    ),
    # Newsletters
    (
        "scv_newsletter_issue_04.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/08/SCV-Newsletter-Issue-04.pdf",
    ),
    (
        "scv_newsletter_issue_05.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/08/News-Letter_Issue-05.pdf",
    ),
    (
        "scv_newsletter_issue_06_aug2023.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2024/01/Newsletter-6-August-2023.pdf",
    ),
    (
        "scv_newsletter_issue_07.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2025/05/Newsletter_Issue_07.pdf",
    ),
    (
        "scv_newsletter_issue_08.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2025/05/Newsletter_Issue_08.pdf",
    ),
    (
        "scv_newsletter_issue_09.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2026/03/Selam-Childrens-Village-Newsletter-Issue-09.pdf",
    ),
    (
        "scv_newsletter_issue_10.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2026/03/Selam-Childrens-Village-Newsletter-Issue-10.pdf",
    ),
    # Annual Reports
    (
        "scv_annual_report_2019.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/09/SCV-Annual-Report-2019.pdf",
    ),
    (
        "scv_annual_report_2020.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/09/SCV-Annual-Report-2020.pdf",
    ),
    (
        "scv_annual_report_2021.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/09/SCV-Annual-Report-2021.pdf",
    ),
    (
        "scv_annual_report_2022.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2023/09/SCV-Annual-Report-2022.pdf",
    ),
    (
        "scv_annual_report_2023.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2024/11/SCV-Annual-Report-2023.pdf",
    ),
    (
        "scv_annual_report_2024.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2025/03/SCV_2024_Annual-Report.pdf",
    ),
    (
        "scv_annual_report_2025.pdf",
        "https://www.selamchildrenvillage.org/wp-content/uploads/2026/08/SCV-2025-Annual-Report.pdf",
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_file(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if not data.startswith(b"%PDF"):
                print(f"  ! {dest.name}: Response was not a valid PDF (length {len(data)})", file=sys.stderr)
                return False
            dest.write_bytes(data)
            return True
    except Exception as exc:
        print(f"  ! Failed downloading {url}: {exc}", file=sys.stderr)
        return False


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {DEST_DIR}")
    print(f"Total publications to process: {len(PUBLICATIONS)}")

    success_count = 0
    total_chars = 0

    for filename, url in PUBLICATIONS:
        dest = DEST_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"✓ Already downloaded: {filename} ({dest.stat().st_size:,} bytes)")
        else:
            print(f"⬇ Downloading: {filename} ...")
            if not download_file(url, dest):
                continue

        # Verify text extraction with pypdf
        try:
            reader = pypdf.PdfReader(str(dest))
            pages = len(reader.pages)
            chars = sum(len(p.extract_text() or "") for p in reader.pages)
            total_chars += chars
            print(f"  └─ Pages: {pages}, Extracted characters: {chars:,}")
            success_count += 1
        except Exception as exc:
            print(f"  ! Error reading PDF {filename}: {exc}")

    print("\n" + "=" * 50)
    print(f"Completed! Successfully downloaded & verified {success_count}/{len(PUBLICATIONS)} publications.")
    print(f"Total text extracted: {total_chars:,} characters.")


if __name__ == "__main__":
    main()
