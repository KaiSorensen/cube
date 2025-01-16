import csv
import os

def saveData(labels, states):
    # Ensure the lengths of labels and states match
    if len(labels) != len(states):
        raise ValueError("Labels and states must have the same length.")

    # if cubedata.csv exists, remove it
    if os.path.exists('cubedata.csv'):
        os.remove('cubedata.csv')

    # Save the data to a CSV file
    with open('cubedata.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write a header for clarity
        writer.writerow(['Label', 'State']) 
        
        # Write each label and its corresponding state
        for label, state in zip(labels, states):
            # state is a list of integers, so join into a comma-separated string
            writer.writerow([label, ','.join(map(str, state))])

def loadData():
    with open('cubedata.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        labels = []
        states = []
        for row in reader:
            labels.append(int(row[0]))
            # Convert the comma-separated string back to a list of integers
            states.append(list(map(int, row[1].split(','))))
        return labels, states