# emotion-aware-interactions-pipeline
A project created for the Human Computer Lab ML Intern role application


#### Issues Faced:

1. Initially Mediapipe FaceMesh didn't work when directly used on the video clips.
Solution: Used FaceDetection first, then Cropped and used FaceMesh for speaker identification.