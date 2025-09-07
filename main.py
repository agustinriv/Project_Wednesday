import pandas as pd

def main():
    print("Inicio de ejecución")

    # Cargar dataset
    df = pd.read_csv("data/data_test.csv")
    print(df.head())
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    with open("logs/logs.txt", "a") as f:
        f.write(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas\n")

if __name__ == "__main__":
    main()

