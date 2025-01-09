import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from fake_useragent import UserAgent
from typing import List, Dict, Optional
import random
import json
import os


class AutoTraderScraper:
    def __init__(self, category: str = "expert"):
        """
        Initialize scraper for either expert reviews or long-term reviews.

        Args:
            category: Either "expert" or "longterm" to specify which type of reviews to scrape
        """
        self.category = category
        if category == "expert":
            self.base_url = "https://www.autotrader.co.uk/content/car-reviews"
            self.url_pattern = "car-reviews"
            self.category_name = "Expert review"
            self.save_directory = "expert_review"
        else:  # longterm
            self.base_url = "https://www.autotrader.co.uk/content/longterm-reviews"
            self.url_pattern = "longterm-reviews"
            self.category_name = "Long-term review"
            self.save_directory = "long_term_reviews"

        self.ua = UserAgent()
        self.session = requests.Session()
        self.processed_urls = set()

        # Setup Selenium
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"user-agent={self.ua.random}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)

    def wait_for_element(self, by, value, timeout=10):
        """Wait for an element to be present on the page"""
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )

    def get_page_content(self, url: str) -> Optional[str]:
        """Fetch page content using Selenium for client-side rendered content"""

        self.driver.get(url)
        self.wait_for_element(By.CSS_SELECTOR, '[data-gui="article-list-container"]')
        time.sleep(5)
        return self.driver.page_source

    def parse_expert_section(self, section: BeautifulSoup) -> Dict:
        """Parse expert review section content."""

        header = section.find(
            ["h2", "h3"],
            class_=["eWczkt", "section-title", "article-section-title"],
        )
        header_text = header.text.strip() if header else None

        content = section.find(
            ["div", "p"],
            class_=["fyJHJB", "section-content", "article-content", "at__sc-21l6gh-3"],
        )
        content_text = content.text.strip() if content else None

        if content_text and not header_text:
            if "news" in str(section.get("class", [])):
                header_text = "News Content"
            elif "expert-review" in str(section.get("class", [])):
                header_text = "Review Content"
            else:
                header_text = "Article Content"

        return {
            "section_title": header_text,
            "content": content_text,
        }

    def parse_longterm_section(self, section: BeautifulSoup) -> Dict:
        """Parse long-term review section content."""

        section_divs = section.find_all("div", class_="at__sc-21l6gh-3")
        if len(section_divs) >= 2:
            title = section_divs[0].get_text(strip=True)
            content = " ".join(div.get_text(strip=True) for div in section_divs[1:])
            return {"section_title": title, "content": content}
        elif len(section_divs) == 1:
            content = section_divs[0].get_text(strip=True)
            return {"content": content}
        return {}

    def parse_article_preview(self, li_element: BeautifulSoup) -> Dict:
        """Parse individual article preview from list item."""

        link = li_element.find("a")
        href = link["href"] if link else None
        if not href or self.url_pattern not in href:
            return {}

        full_url = f"https://www.autotrader.co.uk{href}"
        title = li_element.find("h2", class_="liAhgq")
        title_text = title.text.strip() if title else None

        description = li_element.find("p", class_="iEtmox")
        description_text = description.text.strip() if description else None

        return {
            "title": title_text,
            "description": description_text,
            "url": full_url,
            "category": self.category_name,
        }

    def get_article_content(self, url: str) -> Dict:
        """Fetch and parse article content."""

        self.driver.get(url)
        time.sleep(5)
        content = self.driver.page_source

        soup = BeautifulSoup(content, "html.parser")
        article = soup.find("article")
        if not article:
            return {}

        title = soup.find(["h1", "h2"], class_=["article-title", "at__sc-1n64n0d-2"])
        title_text = title.text.strip() if title else None

        # Extract published date
        published_date = None
        date_container = article.find("div", class_="at__sc-ug5l9f-5")
        if date_container:
            date_element = date_container.find("p", class_="at__sc-1n64n0d-11")
            if date_element:
                date_text = date_element.text.strip()
                if "Published on" in date_text or "Last updated on" in date_text:
                    published_date = (
                        date_text.replace("Published on", "")
                        .replace("Last updated on", "")
                        .split("|")[0]
                        .strip()
                    )

        # Parse sections based on category
        sections = []
        if self.category == "expert":
            article_sections = article.find_all(
                ["div", "section"], class_=["eGyxSU", "at__sc-21l6gh-2"]
            )
            for section in article_sections:
                section_data = self.parse_expert_section(section)
                if section_data:
                    sections.append(section_data)
        else:
            article_sections = article.find_all("section", class_="at__sc-21l6gh-2")
            for section in article_sections:
                section_data = self.parse_longterm_section(section)
                if section_data:
                    sections.append(section_data)

        return {
            "title": title_text,
            "url": url,
            "category": self.category_name,
            "published_date": published_date,
            "sections": sections,
        }

    def scrape_reviews_page(self, page_number: int) -> List[Dict]:
        """Scrape a single page of reviews using Selenium"""
        url = (
            self.base_url if page_number == 1 else f"{self.base_url}?page={page_number}"
        )

        print(f"Fetching URL: {url}")
        content = self.get_page_content(url)
        if not content:
            return []

        soup = BeautifulSoup(content, "html.parser")
        articles_list = soup.find("ul", {"data-gui": "article-list-container"})
        if not articles_list:
            articles_list = soup.find("ul", class_="at__sc-1n64n0d-1")
            if not articles_list:
                return []

        articles = []
        articles_found = False

        for li in articles_list.find_all("li"):
            articles_found = True
            article_preview = self.parse_article_preview(li)
            if article_preview and article_preview["url"] not in self.processed_urls:
                article_content = self.get_article_content(article_preview["url"])
                if article_content:
                    self.processed_urls.add(article_preview["url"])
                    articles.append(article_content)
                    print(f"Processed article: {article_content['title']}")

        if not articles_found:
            return []

        print(f"Found {len(articles)} new articles on page {page_number}")
        return articles

    def scrape_all_reviews(self, max_pages=None):
        """Scrape all review pages up to max_pages"""
        reviews = []
        page = 1
        no_new_content_count = 0

        while True:
            if max_pages and page > max_pages:
                print(f"Reached maximum page limit: {max_pages}")
                break

            print(f"\nScraping page {page}...")
            page_reviews = self.scrape_reviews_page(page)

            if not page_reviews:
                no_new_content_count += 1
                print(f"Pages without new content: {no_new_content_count}")
                if no_new_content_count >= 3:
                    print("Found 3 consecutive pages without new content, stopping")
                    break
            else:
                no_new_content_count = 0
                reviews.extend(page_reviews)

            delay = random.uniform(5, 7)
            print(f"Waiting {delay:.2f} seconds before next page...")
            time.sleep(delay)

            page += 1

        print(f"\nFinished scraping. Total articles collected: {len(reviews)}")
        return reviews

    def save_to_json(self, articles: List[Dict]):
        """Save scraped data to JSON files in organized directory structure."""
        base_dir = os.path.join("articles", "raw")
        os.makedirs(os.path.join(base_dir, self.save_directory), exist_ok=True)

        for article in articles:
            try:
                url_parts = article["url"].rstrip("/").split("/")
                filename = f"{url_parts[-1]}.json"
                file_path = os.path.join(base_dir, self.save_directory, filename)

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)

                print(f"Saved: {file_path}")

            except Exception as e:
                print(f"Error saving article: {e}")


if __name__ == "__main__":
    # For expert reviews
    expert_scraper = AutoTraderScraper(category="expert")
    expert_reviews = expert_scraper.scrape_all_reviews()
    expert_scraper.save_to_json(expert_reviews)

    # For long-term reviews
    longterm_scraper = AutoTraderScraper(category="longterm")
    longterm_reviews = longterm_scraper.scrape_all_reviews()
    longterm_scraper.save_to_json(longterm_reviews)
