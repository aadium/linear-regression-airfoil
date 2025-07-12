from functions import * 

FULL_TRAIN_FILE = './data/full_train.csv'   # filepath of full_train dataset
FULL_TEST_FILE = './data/full_test.csv'     # filepath of full_test dataset

TOY_TRAIN_FILE = './data/toy_train.csv'     # filepath of toy_train dataset
TOY_TEST_FILE = './data/toy_test.csv'       # filepath of toy_test dataset

toyTrainX, toyTrainY = load_data(TOY_TRAIN_FILE)
toyTestX, toyTestY = load_data(TOY_TEST_FILE)
print("===== Toy Dataset =====")
print(f"Toy Train X shape: {toyTrainX.shape}, Toy Train Y shape: {toyTrainY.shape}"
      f"\nToy Test X shape: {toyTestX.shape}, Toy Test Y shape: {toyTestY.shape}")

print("===== Full Dataset =====")
fullTrainX, fullTrainY = load_data(FULL_TRAIN_FILE)
fullTestX, fullTestY = load_data(FULL_TEST_FILE)
print(f"Full Train X shape: {fullTrainX.shape}, Full Train Y shape: {fullTrainY.shape}"
      f"\nFull Test X shape: {fullTestX.shape}, Full Test Y shape: {fullTestY.shape}")

print("========================")
# Design matrices for toy and full datasets
toyTrainX = designMatrix(toyTrainX)
toyTestX = designMatrix(toyTestX)
fullTrainX = designMatrix(fullTrainX)
fullTestX = designMatrix(fullTestX)

# Train and evaluate on different datasets
print("\n" + "="*60)
print("TRAINING AND EVALUATION")
print("="*60)

# Example 1: Full dataset with default parameters
full_theta, full_preds, full_loss = train_and_evaluate(
    fullTrainX, fullTrainY, fullTestX, fullTestY, 
    "Full Dataset", num_epochs=1000, lr=0.01
)