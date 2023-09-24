from django.shortcuts import render
from django.http import JsonResponse
import joblib

# Load the trained model
model = joblib.load('E:\django_project\savedModels\Market_positioning_model.joblib')

# Define the labels for the predictions
labels = {
    0: 'Low',
    1: 'Normal',
    2: 'High',
    3: 'Very High'
}

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        # Get feature values from the form
        battery_power = float(request.POST.get('battery_power'))
        clock_speed = float(request.POST.get('clock_speed'))
        fc = float(request.POST.get('fc'))
        int_memory = float(request.POST.get('int_memory'))
        m_dep = float(request.POST.get('m_dep'))
        mobile_wt = float(request.POST.get('mobile_wt'))
        n_cores = float(request.POST.get('n_cores'))
        pc = float(request.POST.get('pc'))
        px_height = float(request.POST.get('px_height'))
        px_width = float(request.POST.get('px_width'))
        ram = float(request.POST.get('ram'))
        sc_h = float(request.POST.get('sc_h'))
        sc_w = float(request.POST.get('sc_w'))
        talk_time = float(request.POST.get('talk_time'))

        # Make predictions using the loaded model
        features = [battery_power, clock_speed, fc, int_memory, m_dep, mobile_wt, n_cores, pc,
                    px_height, px_width, ram, sc_h, sc_w, talk_time]
        prediction = model.predict([features])[0]

        # Map the numeric prediction to the corresponding label
        prediction_label = labels[prediction]

        return JsonResponse({'prediction': prediction_label})

    return render(request, 'index.html')