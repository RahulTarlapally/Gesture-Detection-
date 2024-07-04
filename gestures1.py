import cv2
import mediapipe as mp

# OpenCV is a library for video capturing and processing
# mediapipe is library for hand tracking and landmark detection

# Step 2: Initializing the MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

#mp_hands: Accesses the hand solution in MediPipe
#hands: Initiliazes the hands module for hand detection
#mp_draw: To draw hand landmarks on the frames

# Step 3: Initializing the video capture
cap = cv2.VideoCapture(0)
# cap: captures video from the default camera (webcam)

# Step 4: Capturing and processing each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally for a later selfie view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Initiliaze list to store landmark coordinates
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get the coordinates
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])
            # code block for gesture recognition
            if len(landmark_list) != 0:
                # example logic for gesture recognition
                # open hand palm gesture
                if landmark_list[4][1] > landmark_list[3][1] and landmark_list[8][1] < landmark_list[6][1]:
                    gesture = "Hello people"
                else:
                    gesture = None
                    #displaying the corresponding text
                    if gesture:
                        cv2.putText(frame, gesture, (landmark_list[0][0] - 50, landmark_list[0][1] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Step 5: Displaying the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.imshow('Hand Gesture Recognition', frame): Displays the frame with hand landmarks
# cv2.waitKey(1): Waits for 1 millisecond for a key press. If 'q' is pressed, the loop breaks

# Step 6: Release resources
cap.release()
cv2.destroyAllWindows()

# cap.release(): Releases the Webcam
# cv2.destroyAllWindows(): Closes all OpenCV windows



