import csv
from pathlib import Path

from google_play_scraper import Sort, reviews


def scrape_play_store_reviews(app_id, jumlah_ulasan, output_dir="data/raw", score=None):
    """
    Fungsi untuk melakukan scraping ulasan dari Google Play Store dan menyimpan ke CSV

    Parameters:
        app_id (str): ID aplikasi (contoh: 'com.ss.android.ugc.trill')
        jumlah_ulasan (int): Jumlah ulasan yang ingin discrape (maksimal 200 per request)
        output_dir (str): Direktori output (default: 'data/raw')
    """
    # Setup path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"reviews_{app_id}_{score}.csv"
    file_path = output_path / filename

    # Lakukan scraping
    try:
        # Gunakan reviews() bukan reviews_all()
        result, continuation_token = reviews(
            app_id,
            lang="id",
            country="id",
            sort=Sort.MOST_RELEVANT,
            count=jumlah_ulasan,
            filter_score_with=score,
        )

        # Format datetime
        def format_date(timestamp):
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Simpan ke CSV
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Review", "Rating", "Tanggal", "Username", "Likes"])

            for review in result[
                :jumlah_ulasan
            ]:  # Pastikan tidak melebihi jumlah_ulasan
                writer.writerow(
                    [
                        review["content"],
                        review["score"],
                        format_date(review["at"]),
                        review["userName"],
                        review["thumbsUpCount"],
                    ]
                )

        print(
            f"Berhasil menyimpan {min(len(result), jumlah_ulasan)} ulasan ke: {file_path.absolute()}"
        )
        return file_path

    except Exception as e:
        print(f"Gagal melakukan scraping: {str(e)}")
        return None


# Contoh penggunaan
if __name__ == "__main__":
    app_id = "com.ss.android.ugc.trill"  # ID aplikasi TikTok
    scrape_play_store_reviews(app_id=app_id, jumlah_ulasan=6000, score=1)
    scrape_play_store_reviews(app_id=app_id, jumlah_ulasan=6000, score=5)
