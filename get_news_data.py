import requests
import pytz
import csv
import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def crawl_thanhnien_article(url):
    """Crawl 1 bài báo Thanh Niên và trả về dict."""
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')

    topic = soup.find('div', class_='detail-cate')
    title = soup.find('h1', class_='detail-title')
    publisher = soup.find('a', class_='name')
    publishedDate = soup.find('div', class_='detail-time')
    description = soup.find('h2', class_='detail-sapo')
    content = soup.find('div', class_='detail-cmain')

    thumbnail_url = None
    if content:
        first_figure = content.find('figure')
        if first_figure:
            a_tag = first_figure.find('a')
            if a_tag and a_tag.has_attr('href'):
                thumbnail_url = urljoin(url, a_tag['href'])
            else:
                img_tag = first_figure.find('img')
                if img_tag and img_tag.has_attr('src'):
                    thumbnail_url = urljoin(url, img_tag['src'])
        for fig in content.find_all('figure'):
            fig.decompose()

    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    crawAt = datetime.now(vietnam_tz)

    match = re.search(r'(\d+)(?=\.htm$)', url)
    article_id = match.group(1) if match else ""

    return {
        'id': article_id,
        'url': url,
        'topic': topic.get_text(strip=True) if topic else "",    
        'publisher': publisher.get_text(strip=True) if publisher else "",
        'publishedDate': publishedDate.get_text(strip=True) if publishedDate else "",
        'crawAt': crawAt,
        'thumbnail': thumbnail_url if thumbnail_url else "",
        'title': title.get_text(strip=True) if title else "",
        'description': description.get_text(strip=True) if description else "",    
        'content': content.get_text(strip=True) if content else ""    
    }

def save_articles_batch(data_list, output_csv):
    """Ghi danh sách bài báo vào CSV."""
    file_exists = os.path.exists(output_csv)
    headers = [
        'id', 'url', 'topic', 'publisher', 'publishedDate',
        'crawAt', 'thumbnail', 'title', 'description', 'content'
    ]
    with open(output_csv, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_list)

# =============================
# 👉 Sử dụng với danh sách URL
# =============================
def get_last_output_url(output_csv: str) -> str | None:
    """Lấy URL ở dòng cuối cùng của file output (bỏ header)."""
    if not os.path.exists(output_csv):
        return None
    last_url = None
    with open(output_csv, newline='', encoding='utf-8-sig') as f:
        # ưu tiên DictReader (vì file có header 'url')
        try:
            reader = csv.DictReader(f)
            for row in reader:
                if row and row.get('url'):
                    last_url = row['url'].strip()
        except Exception:
            f.seek(0)
            reader = csv.reader(f)
            next(reader, None)  # bỏ header nếu có
            for row in reader:
                if row and row[0].strip():
                    last_url = row[0].strip()
    return last_url

def read_all_links(input_file: str) -> list[str]:
    with open(input_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)  # bỏ header
        return [row[0].strip() for row in reader if row and row[0].strip()]

if __name__ == "__main__":
    input_file = 'thanhnien_links.csv'
    output_csv = 'thanhnien_articles.csv'

    all_urls = read_all_links(input_file)
    last_url = get_last_output_url(output_csv)

    # Tìm vị trí của last_url trong file link
    start_idx = -1
    if last_url:
        try:
            start_idx = all_urls.index(last_url)
        except ValueError:
            # không tìm thấy -> coi như crawl từ đầu
            start_idx = -1

    # Lấy danh sách cần crawl từ vị trí +1
    urls = all_urls[start_idx + 1:]
    print(f"🔹 Tổng link: {len(all_urls)} | last_url: {last_url or 'None'} | bắt đầu từ index {start_idx+1} (còn {len(urls)} link).")

    buffer = []
    for idx, url in enumerate(urls, 1):
        print(f"({(start_idx+1)+idx}/{len(all_urls)}) Đang crawl: {url}")
        try:
            article_data = crawl_thanhnien_article(url)
            buffer.append(article_data)
        except Exception as e:
            print(f"⚠️ Lỗi tại {url}: {e}")
        time.sleep(1)

        # lưu batch hoặc lưu nốt khi tới cuối
        if len(buffer) == 10 or ((start_idx+1)+idx) == len(all_urls):
            save_articles_batch(buffer, output_csv)
            print(f"✅ Đã lưu {len(buffer)} bài vào {output_csv}")
            buffer.clear()
