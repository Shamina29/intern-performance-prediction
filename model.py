import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')
le    = joblib.load('label_encoder.pkl')

# Predict function
def predict_performance(completion_time, feedback_rating, attendance):

    # Create features same way as Sprint 3
    performance_index   = feedback_rating * attendance / 100
    completion_speed    = 1 / completion_time
    attendance_category = 0 if attendance < 60 else (1 if attendance < 80 else 2)
    feedback_category   = 0 if feedback_rating <= 2 else (1 if feedback_rating <= 3 else (2 if feedback_rating <= 4 else 3))

    # Put features in correct order
    features = [[
        completion_time,
        feedback_rating,
        attendance,
        performance_index,
        completion_speed,
        attendance_category,
        feedback_category
    ]]

    # Predict label
    prediction = model.predict(features)
    label      = le.inverse_transform(prediction)[0]

    # Get confidence score
    probabilities  = model.predict_proba(features)[0]
    confidence     = round(max(probabilities) * 100, 2)

    # Get probability for each class
    class_probs = {
        le.classes_[i]: str(round(probabilities[i] * 100, 2)) + '%'
        for i in range(len(le.classes_))
    }

    # Give advice based on prediction
    if label == 'High':
        advice = "Excellent performer!"
    elif label == 'Medium':
        advice = "Average performer."
    else:
        advice = "Needs improvement."

    return {
        "prediction"   : label,
        "confidence"   : str(confidence) + '%',
        "probabilities": class_probs,
        "advice"       : advice
    }