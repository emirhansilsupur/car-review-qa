import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from fake_useragent import UserAgent
from typing import Dict, Optional


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
        chrome_options.add_argument("--headless")
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
