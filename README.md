# 528-Project

How to collect data using main.c and read_data.py:

- Choose number of samples to collect by editing the loop in main.c (currently set to collect 33)
- Choose the gesture being collected in read_data.py in lines 6 and 26:
    - 6: OUTPUT_DIR = "../up"
    - 26: filename = os.path.join(OUTPUT_DIR, f"up_{num:02d}.txt")
- build the project
- flash the project
- run read_data.py
- read_data should read the incoming data through serial and seperate it and save it into files

- Running svm_dataset.py extracts the features and saves an X and y file for training
- svm.py runs the SVM algorithm
