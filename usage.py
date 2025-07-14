from lrm import LinearRegressionModel, ClosedFormLinearRegressionModel
import pandas as pd

def load_data(filename: str) -> tuple:
      df = pd.read_csv(filename)
      X = df.iloc[:, 1:].values
      Y = df.iloc[:, 0].values.reshape((-1, 1))
      return (X, Y)

FULL_TRAIN_FILE = './data/full_train.csv'   # filepath of full_train dataset
FULL_TEST_FILE = './data/full_test.csv'     # filepath of full_test dataset

TOY_TRAIN_FILE = './data/toy_train.csv'     # filepath of toy_train dataset
TOY_TEST_FILE = './data/toy_test.csv'       # filepath of toy_test dataset

lrModel = LinearRegressionModel()
cflrModel = ClosedFormLinearRegressionModel()

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
# Train and evaluate on different datasets
print("\n" + "="*60)
print("TRAINING AND EVALUATION")
print("="*60)

full_theta, full_preds, full_loss = lrModel.trainLRModel(
    X_train=fullTrainX, Y_train=fullTrainY, X_test=fullTestX, Y_test=fullTestY, 
    dataset_name="Airfoil Full", num_epochs=1000, lr=0.01
)

closed_theta, closed_preds, closed_loss = cflrModel.trainCLRModel(
    X_train=fullTrainX, Y_train=fullTrainY, X_test=fullTestX, Y_test=fullTestY, 
    dataset_name="Airfoil Full (Closed Form)"
)