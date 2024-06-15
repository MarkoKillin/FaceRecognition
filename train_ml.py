from data_preprocessing import prepare_ml_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from torch.utils.tensorboard import SummaryWriter


def train_model(model, X_train, y_train, X_test, y_test, save_path='models/model_ml.pth'):
    writer = SummaryWriter()

    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print('Model Trained Successfully!')

    #validation



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train machine learning model')
    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'nb', 'rf', 'zr'])
    args = parser.parse_args()

    X, y, classes = prepare_ml_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if args.model == 'dt':
        model = DecisionTreeClassifier()
    elif args.model == 'nb':
        model = GaussianNB()
    elif args.model == 'rf':
        model = RandomForestClassifier()
    else:
        from model_zero_rule import ZeroRuleClassifier
        model = ZeroRuleClassifier()

    train_model(model, X_train, y_train, X_test, y_test, save_path=f'models/model_{args.model}.pkl')
