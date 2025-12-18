import pandas as pd

def dump_2023():
    file_path = "data/raw/Dataset 2023/20230101_power_usage.csv"
    try:
        with open(file_path, "r", encoding="shift_jis") as f:
            content = f.read()
            
        with open("dump_2023.txt", "w", encoding="utf-8") as f:
            f.write(content)
        print("Dumped to dump_2023.txt")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    dump_2023()
