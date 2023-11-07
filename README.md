## KIDS KNOWN GAME ##
using object detection librries, and developing a simple kids game. will require camera usage

--> Use YOLO models/libraries for object detection
    - see the code "windowsfacedetect.py"
--> implement "known person"
    - check if YOLO offers face detection, and re training teh model with known faces
    OR
    - see my code "windowTrainSaveKnownFces.py" for training
    - see my code "windowsVideoDetectKnownFace.py" for testing
        !! face_detection library is old and slow, not suggested, if needed tet the speed of FPS
--> implement front end "known person" to add pictures to train known person
--> implement "text to speech" 
   - see example my code "JN_text2SpeechLibEspeak.py"
      !! this is old code, speech is not suitable for cheerful kids game
## GAME ##
1) Green Go, Red Stop
   track movement, pixel calculation
   track the movement, say the knwon person name, award for the move
3) Rigth Hand, Left hand
   track the announced hands, say the knwon person name, award for the rigth hand
5) Finger Count
   count the fingers,  say the knwon person name, award for the rigth count

## Generic Suggested Functions ##
  - if person is not detected, ask for repositioning
  - use filters not to repeat the same sentences agin and again
