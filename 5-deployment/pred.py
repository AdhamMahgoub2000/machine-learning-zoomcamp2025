import pickle
with open('model.bin','rb') as f_in:
    dv, model = pickle.load(f_in)

X = {
    'lead_source': input("Enter lead source (e.g. paid_ads): "),
    'industry': input("Enter industry (or NaN): "),
    'number_of_courses_viewed': int(input("Enter number of courses viewed: ")),
    'annual_income': float(input("Enter annual income: ")),
    'employment_status': input("Enter employment status (e.g. unemployed): "),
    'location': input("Enter location (e.g. south_america): "),
    'interaction_count': int(input("Enter interaction count: ")),
    'lead_score': float(input("Enter lead score: "))
}

X_vectorized = dv.transform([X])
y_pred = model.predict_proba(X_vectorized)[:, 1]
print(f'Predicted probability of conversion: {y_pred[0]}')