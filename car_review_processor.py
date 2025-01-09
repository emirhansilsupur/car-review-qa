import os
from pathlib import Path
import json
import re
from typing import Dict, List
from tqdm import tqdm


class CarReviewDataProcessor:
    def __init__(self):
        self.make_types = [
            "abart",
            "abarth",
            "alpine",
            "ariel",
            "audi",
            "bmw",
            "byd",
            "bentley",
            "bugatti",
            "cupra",
            "caterham",
            "chevrolet",
            "chrysler",
            "citroen",
            "ds",
            "dacia",
            "dodge",
            "ferrari",
            "fiat",
            "fisker",
            "ford",
            "genesis",
            "honda",
            "hyundai",
            "ineos",
            "infiniti",
            "isuzu",
            "jaguar",
            "jeep",
            "kia",
            "lamborghini",
            "leapmotor",
            "lexus",
            "lotus",
            "maserati",
            "mazda",
            "mclaren",
            "mercedes",
            "mg",
            "micro",
            "mini",
            "mitsubishi",
            "nio",
            "nissan",
            "omoda",
            "perodua",
            "peugeot",
            "polestar",
            "porsche",
            "proton",
            "renault",
            "saab",
            "seat",
            "skoda",
            "skywell",
            "smart",
            "ssangyong",
            "subaru",
            "suzuki",
            "tesla",
            "toyota",
            "vauxhall",
            "volkswagen",
            "volvo",
            "xpeng",
            "zeekr",
        ]

        self.make_types_w_ = [
            "mercedes-benz",
            "alfa-romeo",
            "aston-martin",
            "ds-automobiles",
            "gwm-ora",
            "land-rover",
            "range-rover",
            "rolls-royce",
        ]

        self.exclude_words = [
            "review",
            "reviews",
            "test",
            "drive",
            "preview",
            "long",
            "term",
            "final",
            "report",
            "second",
            "third",
            "fourth",
            "fifth",
            "first",
            "edition",
            "vignale",
            "expert",
        ]

        self.body_types = [
            "hatchback",
            "estate",
            "saloon",
            "suv",
            "coupe",
            "convertible",
            "mpv",
            "pickup",
            "4x4",
            "hybrid",
            "electric",
            "hatch",
            "sport",
        ]

    def parse_car_filename(self, filename: str, base_directory: str = None) -> Dict:
        """Parse car review filename into structured data."""
        if isinstance(filename, Path):
            filename = str(filename)

        # Clean the input path to just the filename
        filename = filename.replace("\\", "/").split("/")[-1]

        # Remove .json extension and review-type suffixes
        base = filename.replace(".json", "")
        base = base.replace("-expert-review", "").replace("-long-term-test-review", "")

        # Handle "living with" format for long-term reviews
        if base.startswith("living-with-a-"):
            base = base.replace("living-with-a-", "", 1)
        elif base.startswith("living-with-an-"):
            base = base.replace("living-with-an-", "", 1)

        # Split remaining parts
        parts = base.split("-")

        # Attempt to extract make
        make = None
        # First try multi-word makes
        for length in range(2, 0, -1):
            if len(parts) >= length:
                candidate = "-".join(parts[:length]).lower()
                if candidate in self.make_types_w_:
                    make = candidate
                    parts = parts[length:]
                    break

        # If no multi-word make found, try single word makes
        if make is None and parts:
            candidate = parts[0].lower()
            if candidate in self.make_types:
                make = candidate
                parts = parts[1:]

        # Find model parts
        model_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower not in self.exclude_words:
                model_parts.append(part)

        # Join model parts
        model = "-".join(model_parts).lower() if model_parts else None

        # For long term reviews directory, return simplified structure
        if (
            base_directory
            and base_directory.strip("/") == "articles/raw/long_term_reviews"
        ):
            return {"make": make, "model": model}

        # Extract additional metadata for expert reviews
        year_match = re.search(r"-(\d{4})(?:-|$)", base)
        year = year_match.group(1) if year_match else None

        # Find body type components
        body_type_parts = []
        for part in parts:
            if part.lower() in self.body_types:
                body_type_parts.append(part.lower())

        body_type = "-".join(body_type_parts) if body_type_parts else None

        return {"make": make, "model": model, "body_type": body_type, "year": year}

    def process_review_file(self, file_path: str, base_directory: str) -> bool:
        """Process a single review file and add extracted metadata."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get the filename from the path
            filename = os.path.basename(file_path)

            # Extract metadata from filename
            metadata = self.parse_car_filename(filename, base_directory)

            # Add file path information
            data["car_details"] = {**metadata}

            # Write back the updated JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return False

    def process_directory(self, directory: str) -> List[str]:
        """Process all JSON files in a directory."""
        base_path = Path(directory)
        json_files = list(base_path.glob("**/*.json"))
        processed_files = []

        for file_path in tqdm(json_files, desc=f"Processing files in {directory}"):
            if self.process_review_file(file_path, directory):
                processed_files.append(str(file_path))

        return processed_files


def main():
    processor = CarReviewDataProcessor()

    # Process expert reviews
    expert_dir = "articles/raw/expert_review"
    if os.path.exists(expert_dir):
        print("\nProcessing expert reviews...")
        processed_expert = processor.process_directory(expert_dir)
        print(f"Processed {len(processed_expert)} expert review files")

    # Process long-term reviews
    longterm_dir = "articles/raw/long_term_reviews"
    if os.path.exists(longterm_dir):
        print("\nProcessing long-term reviews...")
        processed_longterm = processor.process_directory(longterm_dir)
        print(f"Processed {len(processed_longterm)} long-term review files")


if __name__ == "__main__":
    main()
