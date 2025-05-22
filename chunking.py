import re
import os
import requests
import json

def download_google_drive_json(share_url, output_folder="downloads"):
    os.makedirs(output_folder, exist_ok=True)
    file_id_match = re.search(r"/d/([^/]+)", share_url)
    if not file_id_match:
        raise ValueError(f"Invalid Google Drive URL: {share_url}")
    file_id = file_id_match.group(1)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    return response.content

def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")

def extract_year(date_str):
    match = re.match(r"(\d{4})-\d{2}-\d{2}", date_str)
    if match:
        return int(match.group(1))
    return 1947

def chunk_constitution_json(json_data):
    title = json_data.get("title", "Constitution")
    print(title)
    slug = slugify(title)
    year = extract_year(json_data.get("date", "1947-01-01"))

    chunks = []

    # Handle preamble
    if "preamble" in json_data:
        chunks.append({
            "text": json_data["preamble"],
            "metadata": {
                "title": title,
                "slug": slug,
                "section": "Preamble",
                "chapter": None,
                "article": None,
                "year": year
            }
        })
    
    # Handle articles (additional articles)
    for article in json_data.get("articles", []):
        content = article.get("content")
        number = article.get("number")
        chunks.append({
            "text": content,
            "metadata": {
                "title": title,
                "slug": slug,
                "section": None,
                "chapter": None,
                "article": str(number),
                "year": year
            }
        })

    # Handle chapters (main text)
    for chapter in json_data.get("chapters", []):
        chapter_number = chapter.get("number")
        chapter_title = chapter.get("title", f"Chapter {chapter_number}")
        chapter_label = f"Chapter {chapter_number}: {chapter_title.strip()}"

        # If there are sections
        for section in chapter.get("sections", []):
            section_title = section.get("title")
            for article in section.get("articles", []):
                content = article.get("content")
                number = article.get("number")
                chunks.append({
                    "text": content,
                    "metadata": {
                        "title": title,
                        "slug": slug,
                        "section": section_title or chapter_label,
                        "chapter": chapter_label,
                        "article": str(number),
                        "year": year
                    }
                })

        # Or if there are direct articles under chapter
        for article in chapter.get("articles", []):
            content = article.get("content")
            number = article.get("number")
            chunks.append({
                "text": content,
                "metadata": {
                    "title": title,
                    "slug": slug,
                    "section": None,
                    "chapter": chapter_label,
                    "article": str(number),
                    "year": year
                }
            })

    return chunks

def process_json_files(json_links, output_path="chunks.json"):
    all_chunks = []
    for link in json_links:
        print(f"Processing: {link}")
        try:
            json_bytes = download_google_drive_json(link)
            json_content = json.loads(json_bytes)
            chunks = chunk_constitution_json(json_content)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ Failed to process {link}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    json_links = [
        "https://drive.google.com/file/d/1proka9W9ygNNVSnuCyCiLBygJoEgdSat/view?usp=sharing",
        "https://drive.google.com/file/d/1rrxapBAxwhUBEOPe2lpyZf9bWYuKBISM/view?usp=sharing"
    ]
    process_json_files(json_links, output_path="chunks.json")
