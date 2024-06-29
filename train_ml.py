from data_preprocesing import prepare_ml_data_lfw, prepare_ml_data_cwf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from model_zero_rule import ZeroRuleClassifier
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib


def train_model(model, X_train, y_train, X_test, y_test, save_path='models/model_ml.pth', log_path='runs/'):
    writer = SummaryWriter(log_dir=log_path)

    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print('Model Trained Successfully!')

    # validation
    y_pred = model.predict(X_test)
    val_accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {val_accuracy}')
    writer.add_scalar('Accuracy', val_accuracy, 0)

    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train machine learning model')
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'nb', 'zr'])
    args = parser.parse_args()

    X, y, h, w = prepare_ml_data_cwf()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_components = 150
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('----Data Loaded----')

    if args.model == 'svm':  # polinomial kernel degree 1
        # param_grid_svm = {'C': [1e3, 1e4, 1e5], 'gamma': [0.0001, 0.001, 0.01, 0.1], }
        # model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid_svm)
        model = SVC(kernel='rbf', random_state=42)
    elif args.model == 'nb':
        model = GaussianNB()
    else:
        model = ZeroRuleClassifier()

    train_model(model, X_train_pca, y_train, X_test_pca, y_test, save_path=f'models/model_{args.model}.pkl',
                log_path=f'runs/model_{args.model}')
