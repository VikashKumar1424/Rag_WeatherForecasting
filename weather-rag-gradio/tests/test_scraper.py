
from weather_rag.scraper import OpenMeteoScraper


def test_scraper_has_expected_methods():
    scraper = OpenMeteoScraper()
    assert callable(scraper.geocode)
    assert callable(scraper.fetch_forecast)
    assert callable(scraper.scrape_docs_metadata)
