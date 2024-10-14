import pandas as pd
import numpy as np

def process_seismic_data(csv_data):
    # Convert the CSV data to a pandas DataFrame
    dataset = pd.read_csv(csv_data)

    # Filter and process the seismic data using the mathematical core
    filtered_data = apply_filter(dataset)
    events = detect_seismic_events(filtered_data)

    # Return the results as a dictionary
    return {'filtered_data': filtered_data.to_dict(), 'events': events.to_dict()}

def estimatePos(rVel, time):
    expecPos = np.empty(len(time))
    expecPos[0] = 0
    for i in range(1, len(time)):
        deltaTime = time[i] - time[i-1]
        acc = (rVel[i] - rVel[i-1]) / deltaTime
        expecPos[i] = expecPos[i-1] + rVel[i-1] * (deltaTime + (0.5 * acc * deltaTime**2))

    return expecPos

def estimateVel(rVel, time):
    expecVel = np.empty(len(time))
    expecVel[0] = 0
    for i in range(1, len(time)):
        deltaTime = time[i] - time[i-1]
        acc = (rVel[i] - rVel[i-1]) / deltaTime
        expecVel[i] = rVel[i-1] + acc * deltaTime

    return expecVel

def estimateNoise(rVel, expecVel):
    obsNoise = np.empty(len(rVel))
    for i in range(len(rVel)):
        obsNoise[i] = rVel[i] - expecVel[i]

    return obsNoise

def maxAcc(rVel, time):
    maxAcc = 0
    for i in range(1, len(time)):
        deltaTime = time[i] - time[i-1]
        acc = (rVel[i] - rVel[i-1]) / deltaTime
        if acc > maxAcc:
            maxAcc = acc

    return maxAcc

def calcMeasurementNoise(rVel, expecVel):
    R  =0
    for i in range(len(rVel)):
        R += (rVel[i] - expecVel[i])**2

    return R/(len(rVel) - 1)

def kalmanFilter(rVel, time):
    # Use independent vectors as state matrix
    x = estimatePos(rVel, time)
    v = estimateVel(rVel, time)

    # Calculate observed noise
    w = estimateNoise(rVel, expecVel)

    # Define Error Covariance Matrix
    Px = np.empty(len(time))
    Pv = np.empty(len(time))
    Px[0] = 0
    Pv[0] = 0

    # Define Predicted Error Covariance Matrix
    PxP = np.empty(len(time))
    PvP = np.empty(len(time))
    PxP[0] = 0
    PvP[0] = 0

    # Define yk
    yx = np.empty(len(time))
    yv = np.empty(len(time))
    yx[0] = 0
    yv[0] = 0

    # Calculate Measurement Noise
    R = calcMeasurementNoise(rVel, expecVel)

    # Define Kalman Gain
    Kx = np.empty(len(time))
    Kv = np.empty(len(time))
    Kx[0] = 0
    Kv[0] = 0

    for i in range(1, len(time)):
        
        deltaTime = time[i] - time[i-1]
        
        # Calculate Noise Process
        Q = (maxAcc(rVel, time)**2) * deltaTime**2

        # Prediction Step
        xP = x[i-1] + v[i-1] * deltaTime
        vP = v[i-1]

        # Predict Error Covariance Matrix
        PxP[i] = (Px[i - 1] + Pv[i - 1]) + (Px[i - 1] + Pv[i - 1]) + Q
        PvP[i] = (deltaTime * Pv[i - 1]) + Pv[i - 1] + Q

        # Update Step
        yx[i] = 0
        yv[i] = rVel[i] - vP

        # Compute Kalman Gain
        Kx[i] = 0
        Kv[i] = PvP[i] / (PvP[i] + R)

        # Update State Matrix
        x[i] = xP
        v[i] = vP + Kv[i] * yv[i]

        # Update Error Covariance Matrix
        Px[i] = 0
        Pv[i] = (1 - Kv[i]) * PvP[i]
    
    return x, v

def apply_filter(dataset):
    # Extract the relevant columns from the dataset
    relTime = np.array(dataset["rel_time(sec)"].to_list())
    obsVel = np.array(dataset["velocity(c/s)"].to_list())
    absTime = np.array(dataset["time(%Y-%m-%dT%H:%M:%S.%f)"].to_list())

    # Apply the Kalman filter
    filteredPos, filteredVel = kalmanFilter(obsVel, relTime)

    # Save the filtered position and velocity in a csv file with the following columns: relative time, velocity and position
    filteredData = pd.DataFrame({"absolute time": absTime, "relative time": relTime, "velocity": filteredVel, "position": filteredPos})

    return filteredData

def detect_seismic_events(filtered_data):
    # extract the relevant columns from the filtered data
    filteredPos = np.array(filtered_data["position"].to_list())
    filteredVel = np.array(filtered_data["velocity"].to_list())
    absTime = np.array(filtered_data["absolute time"].to_list())

    # Compute the mean and standard deviation of the filtered velocity
    meanVel = np.mean(filteredVel)
    stdVel = np.std(filteredVel)

    # Compute the threshold
    threshold = meanVel + 8.7 * stdVel

    # Detect seismic events in the data and catalogue the in csv file with the following columns: id, abolute time, velocity and position
    # Save the data in a csv file
    events = []
    for i in range(1, len(filteredVel)):
        if filteredVel[i] > threshold:
            # Find the absolute time of the event in the original data
            abskTime = absTime.iloc[i]

            events.append([i, abskTime, filteredVel[i], filteredPos[i]])

    events = pd.DataFrame(events, columns=["id", "abolute time", "velocity", "position"])

    return events