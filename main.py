from src.config import DATA_PATH
from src.data_loader import load_data


def main():
    df = load_data(DATA_PATH)

    print("Data Loaded Successfully")
    print(df.head())


if __name__ == "__main__":
    main()
