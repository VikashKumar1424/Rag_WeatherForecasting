
from __future__ import annotations

import argparse

from .rag import build_vector_store, weather_documents
from .scraper import OpenMeteoScraper


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the weather RAG index.")
    parser.add_argument("--location", required=True, help="Any Indian city, district, or state")
    parser.add_argument("--country", default="India", help="Keep this as India for this application")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--forecast-days", type=int, default=7)
    args = parser.parse_args()

    scraper = OpenMeteoScraper()

    if args.latitude is None or args.longitude is None:
        geo = scraper.geocode(args.location, args.country)
        latitude = float(geo["latitude"])
        longitude = float(geo["longitude"])
        location_name = geo.get("name", args.location)
        timezone = geo.get("timezone", "auto")
    else:
        latitude = args.latitude
        longitude = args.longitude
        location_name = args.location
        timezone = "auto"

    forecast = scraper.fetch_forecast(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        forecast_days=args.forecast_days,
    )

    docs_metadata = scraper.scrape_docs_metadata()
    documents = weather_documents(
        forecast=forecast,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
    )

    for doc in documents:
        doc.metadata["docs_title"] = docs_metadata["title"]
        doc.metadata["docs_url"] = docs_metadata["source_url"]

    build_vector_store(documents)

    print(f"Indexed {len(documents)} hourly weather records for {location_name}.")
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"Docs source: {docs_metadata['source_url']}")


if __name__ == "__main__":
    main()
