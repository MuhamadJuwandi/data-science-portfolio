import pandas as pd
import glob

def inspect_2023():
    file_path = "data/raw/Dataset 2023/20230101_power_usage.csv"
    print(f"Inspecting {file_path}")
    
    encodings = ['shift_jis', 'cp932']
    
    for enc in encodings:
        print(f"\n--- Trying encoding: {enc} ---")
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print("Success reading!")
            
            with open("inspection_result.txt", "w", encoding="utf-8") as f:
                f.write(f"Encoding: {enc}\n")
                f.write(f"Columns: {df.columns.tolist()}\n")
                f.write(df.head().to_string())
            print("Written to inspection_result.txt")
            break
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    inspect_2023()
