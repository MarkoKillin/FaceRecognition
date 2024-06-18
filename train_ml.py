from data_preprocesing import prepare_ml_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import joblib


def train_model(model, X_train, y_train, X_test, y_test, save_path='models/model_ml.pth', log_path='runs/'):
    writer = SummaryWriter(log_dir=log_path)

    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print('Model Trained Successfully!')

    # validation
    y_pred = model.predict(X_test)
    print('Predicted')
    val_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy')
    writer.add_scalar('Accuracy', val_accuracy, 0)
    print('written')

    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train machine learning model')
    parser.add_argument('--model', type=str, default='rf', choices=['svm', 'nb', 'rf', 'zr'])
    args = parser.parse_args()

    X, y, classes = prepare_ml_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('----Data Loaded----')

    if args.model == 'svm':
        model = SVC(kernel=linear)
    elif args.model == 'nb':
        model = GaussianNB()
    elif args.model == 'rf':
        model = RandomForestClassifier()
    else:
        from model_zero_rule import ZeroRuleClassifier

        model = ZeroRuleClassifier()

    train_model(model, X_train, y_train, X_test, y_test, save_path=f'models/model_{args.model}.pkl',
                log_path=f'runs/model_{args.model}')
