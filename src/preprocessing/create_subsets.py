from sklearn.model_selection import train_test_split

def create_stratified_subsets(X,y, train_ratio, test_ratio, verify_ratio):

    # Split into train and temp, keeping class ratios intact
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_ratio+verify_ratio, stratify=y, random_state=42
    )

    # Split temp into test and verify, maintaining class ratios
    X_test, X_verify, y_test, y_verify = train_test_split(
        X_temp, y_temp, test_size=(test_ratio / (test_ratio + verify_ratio)), stratify=y_temp, random_state=42
    )

    return X_train, X_test, X_verify, y_train, y_test, y_verify