def build_model():
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    import joblib
    print("Loading data...")
    df = pd.read_csv('houses.csv')
    print(f"Data loaded: {len(df)} rows")
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    print("Training model...")
    model = LinearRegression()
    model.fit(X, y)
    print("Model trained. Saving...")
    joblib.dump(model, "regression.joblib")
    print("Model saved successfully!")
    
build_model()