# -*- coding: utf-8 -*-
import os, csv, time
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


def crawl_infinite_links(
    category_url: str,
    css_selector: str = 'a.box-category-link-with-avatar.img-resize',
    output_csv: str = 'thanhnien_links.csv',
    max_scrolls: int = 60,
    pause_sec: float = 1.2,
    stop_if_no_new_rounds: int = 3,
    headless: bool = True,
    try_click_selectors: tuple = (
        'a.view-more.btn-viewmore[href="javascript:;"]',
    ),
):
    """
    Cuộn vô hạn và thu thập tất cả href theo css_selector từ category_url.
    Lưu CSV (append, không ghi đè, bỏ qua link trùng). Có phát hiện 'kẹt cuộn'.
    """
    # --- Chrome Options (giảm bị chặn headless) ---
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1280,2600")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option('useAutomationExtension', False)
    chrome_opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_opts
    )

    # --- Load links cũ (tránh trùng) ---
    existing = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row['url'])
        print(f"🔹 Loaded {len(existing)} existing links from {output_csv}")

    collected = set()
    no_new_rounds = 0

    try:
        driver.get(category_url)
        time.sleep(2)

        # Helper: đóng cookie/consent nếu có (best-effort)
        for sel in ['button#onetrust-accept-btn-handler', '.ot-pc-refuse-all-handler',
                    'button[aria-label="accept"]', 'button:contains("Đồng ý")']:
            try:
                for el in driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed() and el.is_enabled():
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.5)
            except Exception:
                pass

        # Track chiều cao trang để biết có còn load tiếp không
        last_height = driver.execute_script("return document.body.scrollHeight")

        for i in range(max_scrolls):
            # 1) Thử click 'xem thêm' nếu có
            for sel in try_click_selectors:
                try:
                    for btn in driver.find_elements(By.CSS_SELECTOR, sel):
                        if btn.is_displayed() and btn.is_enabled():
                            try:
                                driver.execute_script("arguments[0].click();", btn)
                                time.sleep(pause_sec)
                            except WebDriverException:
                                try:
                                    btn.click(); time.sleep(pause_sec)
                                except Exception:
                                    pass
                except Exception:
                    pass

            # 2) Cuộn từng bước nhỏ để kích hoạt lazy load theo viewport
            for _ in range(5):
                driver.execute_script(
                    "window.scrollBy(0, Math.floor(window.innerHeight*0.9));"
                )
                time.sleep(pause_sec/3)

            # 3) Cuộn tới đáy + chờ
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_sec)

            # 4) Kiểm tra chiều cao trang có tăng không
            new_height = driver.execute_script("return document.body.scrollHeight")
            height_grew = new_height > last_height
            last_height = new_height

            # 5) Parse & gom link
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            before = len(collected)
            for a in soup.select(css_selector):
                href = a.get('href')
                if not href:
                    continue
                full = urljoin(category_url, href)
                if (full not in existing) and (full not in collected):
                    collected.add(full)
            new_links = len(collected) - before

            print(f"🔹 Round {i+1}: +{new_links} new, total={len(existing) + len(collected)} | height_grew={height_grew}")

            # 6) Logic dừng: không có link mới trong vòng này **hoặc** chiều cao không tăng 3 vòng liên tiếp
            if new_links == 0 and not height_grew:
                no_new_rounds += 1
                if no_new_rounds >= stop_if_no_new_rounds:
                    print(f"ℹ️  Stop: no progress for {no_new_rounds} rounds.")
                    break
            else:
                no_new_rounds = 0

        # --- Lưu CSV (append, chỉ link mới) ---
        file_exists = os.path.exists(output_csv)
        with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writerow(['url'])
            # Ghi theo set difference, có sort để ổn định
            new_only_sorted = sorted(collected)
            for link in new_only_sorted:
                writer.writerow([link])

        print(f"✅ Added {len(new_only_sorted)} new links. "
              f"Total (existing+new): {len(existing) + len(new_only_sorted)}. Saved to {output_csv}")

    finally:
        driver.quit()

def crawl_infinite_links_2(
    category_url: str,
    css_selector: str = 'a.box-category-link-with-avatar.img-resize',
    output_csv: str = 'thanhnien_links.csv',
    max_scrolls: int = 60,          # vẫn giữ cho trường hợp không dùng thời gian
    pause_sec: float = 1.2,
    stop_if_no_new_rounds: int = 3,
    headless: bool = True,
    try_click_selectors: tuple = (
        'a.view-more.btn-viewmore[href="javascript:;"]',
    ),
    # >>> THÊM 3 THAM SỐ MỚI <<<
    max_minutes: float | None = None,    # nếu set (vd 5.0) -> chạy theo thời gian
    idle_timeout_secs: int = 90,         # không có link mới quá X giây -> dừng
    save_every_secs: int = 60,           # ghi CSV tạm mỗi X giây
):
    # --- Chrome Options (giảm bị chặn headless) ---
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1280,2600")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option('useAutomationExtension', False)
    chrome_opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_opts
    )

    # --- Load links cũ (tránh trùng) ---
    existing = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row['url'])
        print(f"🔹 Loaded {len(existing)} existing links from {output_csv}")

    collected = set()
    no_new_rounds = 0

    # >>> THÊM: biến theo dõi thời gian <<<
    start_ts = time.time()
    last_new_ts = start_ts
    last_save_ts = start_ts

    try:
        driver.get(category_url)
        time.sleep(2)

        # đóng consent (giữ nguyên)
        for sel in ['button#onetrust-accept-btn-handler', '.ot-pc-refuse-all-handler',
                    'button[aria-label="accept"]', 'button:contains("Đồng ý")']:
            try:
                for el in driver.find_elements(By.CSS_SELECTOR, sel):
                    if el.is_displayed() and el.is_enabled():
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.5)
            except Exception:
                pass

        last_height = driver.execute_script("return document.body.scrollHeight")

        # >>> NẾU CÓ max_minutes -> dùng while theo thời gian; ngược lại dùng for như cũ <<<
        round_idx = 0
        def time_left():
            return (max_minutes is None) or (time.time() - start_ts < max_minutes * 60)

        while time_left() if max_minutes is not None else round_idx < max_scrolls:
            round_idx += 1

            # 1) click nút xem thêm (nếu có)
            for sel in try_click_selectors:
                try:
                    for btn in driver.find_elements(By.CSS_SELECTOR, sel):
                        if btn.is_displayed() and btn.is_enabled():
                            try:
                                driver.execute_script("arguments[0].click();", btn)
                                time.sleep(pause_sec)
                            except WebDriverException:
                                try:
                                    btn.click(); time.sleep(pause_sec)
                                except Exception:
                                    pass
                except Exception:
                    pass

            # 2) Cuộn kích hoạt lazy-load
            for _ in range(5):
                driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.9));")
                time.sleep(pause_sec/3)

            # 3) Cuộn tới đáy
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_sec)

            # 4) Kiểm tra tăng chiều cao
            new_height = driver.execute_script("return document.body.scrollHeight")
            height_grew = new_height > last_height
            last_height = new_height

            # 5) Parse & gom link
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            before = len(collected)
            for a in soup.select(css_selector):
                href = a.get('href')
                if not href:
                    continue
                full = urljoin(category_url, href)                
                if (full not in existing) and (full not in collected):
                    collected.add(full)
            new_links = len(collected) - before

            # cập nhật “có tiến triển”
            if new_links > 0:
                last_new_ts = time.time()
                no_new_rounds = 0
            else:
                no_new_rounds += 1

            print(f"🔹 Round {round_idx}: +{new_links} new, total={len(existing) + len(collected)} | height_grew={height_grew}")

            # 6) Ghi CSV tạm theo chu kỳ (tuỳ chọn)
            now = time.time()
            if save_every_secs and (now - last_save_ts >= save_every_secs) and len(collected) > 0:
                file_exists = os.path.exists(output_csv)
                with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    if not file_exists:
                        writer.writerow(['url'])
                    for link in sorted(collected):
                        writer.writerow([link])
                existing.update(collected)
                collected.clear()
                last_save_ts = now
                print(f"💾 Saved interim batch. Total saved so far: {len(existing)}")

            # 7) Điều kiện dừng
            if max_minutes is None:
                # chế độ cũ: dừng theo vòng + chiều cao
                if new_links == 0 and not height_grew:
                    if no_new_rounds >= stop_if_no_new_rounds:
                        print(f"ℹ️  Stop: no progress for {no_new_rounds} rounds.")
                        break
            else:
                # chế độ theo thời gian: nếu không có link mới quá idle_timeout_secs -> dừng
                if now - last_new_ts >= idle_timeout_secs:
                    print(f"ℹ️  Stop: idle {int(now - last_new_ts)}s without new links (time-mode).")
                    break

        # --- Lưu phần còn lại (nếu có) ---
        if len(collected) > 0:
            file_exists = os.path.exists(output_csv)
            with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writerow(['url'])
                for link in sorted(collected):
                    writer.writerow([link])

        print(f"✅ Done. Appended new links. Check {output_csv}")

    finally:
        driver.quit()



# ============================
# 👉 Cách sử dụng
# ============================
if __name__ == "__main__":
    # # Danh sách chuyên mục cần crawl
    # categories = [
    #     # "https://thanhnien.vn/kinh-te.htm",
    #     # "https://thanhnien.vn/thoi-su/phap-luat.htm",
    #     "https://thanhnien.vn/the-thao.htm",
    # ]

    # # Chạy lần lượt từng chuyên mục
    # for url in categories:
    #     print(f"\n🌐 Crawling category: {url}")
    #     crawl_infinite_links(
    #         category_url=url,
    #         css_selector='a.box-category-link-with-avatar',
    #         output_csv='thanhnien_links.csv',
    #         max_scrolls=20,
    #         pause_sec=1.2,
    #         stop_if_no_new_rounds=5,
    #         headless=False
    #     )
       
    crawl_infinite_links_2(
        category_url="https://thanhnien.vn/the-thao.htm",
        css_selector="a.box-category-link-title",
        output_csv="thanhnien_links.csv",
        headless=False,
        max_minutes=15.0,          # ⬅️ chạy đúng 15 phút (trừ khi idle quá lâu)
        idle_timeout_secs=600,     # không có link mới > 600s thì thoát sớm
        save_every_secs=60,        # ghi tạm mỗi 60s        
    )