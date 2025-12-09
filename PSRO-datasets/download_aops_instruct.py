
from datasets import load_dataset

def download_aops_instruct():
    print("Downloading AoPS-Instruct dataset...")
    dataset = load_dataset("DeepStudentLlama/AoPS-Instruct", split="train")

    print("Saving dataset as CSV file...")
    dataset.to_pandas().to_csv("aops_instruct_train.csv", index=False)

    print("Dataset downloaded and saved as aops_instruct_train.csv")


if __name__ == "__main__":
    download_aops_instruct()
