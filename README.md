# emotion-aware-interactions-pipeline
A project created for the Human Computer Lab ML Intern role application


#### Issues Faced:

1. Initially Mediapipe FaceMesh didn't work when directly used on the video clips.
Solution: Used FaceDetection first, then Cropped and used FaceMesh for speaker identification.

2. Had problems with shape mismatch of 3 visual frames but embeddings requiring singular linear data
Solution: Flattened vision tokens as well in a similar fashion to text tokens3. 

3. Faced Issues with BFloat16 and Float mismatch between emo head output, and vision tokens being in float but the LLM being in BFloat16
Solution: Custom Casting

4. Had to remove one clip Diag 125 utt 3 due to file corruption

5. Training took too long due to CLIP features being computed during training loop. 
Solution: precomputed for the whole dataset