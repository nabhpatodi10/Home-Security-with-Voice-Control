# Home-Security-with-Voice-Control

## Repository Directory
```plaintext
├── Face_Dataset/            # Face Detection Dataset
├── Weapon_Dataset/          # Weapon Detection Dataset
├── Voice_Assistant/         # Entire Voice Assistant Code
│   ├── db/                  # Database Connection for Voice Assistant
|       └── db.py            
│   ├── langchainget/        # Query file
|       └── query.py
|   ├── Voice_Assistant.py   # Code for running the Voice Assistant
├── Content.txt              # Content file for Voice Assistant
├── .env                     # All APIs used in the project
├── face_detection.py        # Face Identification Model Training file  
├── Model.h5                 # Face Identification model file
├── test_face.py             # Face Detection and Identification Testing
├── yolo.py                  # Face Detection using Yolo File
├── train_yolo.py            # Fine Tuning Yolo for Weapons File
├── yolo11n.pt               # Yolo pretrained model - not fine tuned
├── yolo11x.pt               # Yolo pretrained model - not fine tuned
├── pipeline.py              # Final Pipeline of the project
└── requirements.txt         # Requirements file
```

### Note:
It is suggested to use a virtual environment for this project
