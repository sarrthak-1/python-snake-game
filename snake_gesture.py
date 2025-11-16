"""
Snake game (Pygame) + index finger gesture controls
Python 3.10 compatible

Gestures:
  - Fist → Pause
  - Open palm → Play
  - Thumbs-down → Quit
Movement:
  - Index finger position → Up/Down/Left/Right
"""

import cv2
import mediapipe as mp
import pygame
import random
import threading
import time
from collections import deque

# ===========================
# HAND GESTURE DETECTOR
# ===========================
class GestureDetector(threading.Thread):
    def __init__(self, cam_index=0):
        threading.Thread.__init__(self, daemon=True)
        self.cap = cv2.VideoCapture(cam_index)
        self.running = True

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

        self.lock = threading.Lock()
        self.gesture = "play"
        self.direction = None
        self.hist = deque(maxlen=7)
        self.quit_hold = None
        self.frame = None

    def get_gesture(self):
        with self.lock:
            return self.gesture

    def get_direction(self):
        with self.lock:
            return self.direction

    def count_fingers(self, lm):
        fingers = []
        try:
            fingers.append(int(lm[4].x < lm[3].x))  # Thumb
        except:
            fingers.append(0)
        for tid in [8, 12, 16, 20]:
            try:
                fingers.append(int(lm[tid].y < lm[tid-2].y))
            except:
                fingers.append(0)
        return fingers  # [thumb, index, middle, ring, pinky]

    def is_thumbs_down(self, lm):
        wrist, tip, ip = lm[0], lm[4], lm[3]
        thumb_extended = abs(tip.x - ip.x) > 0.03
        thumb_down = (tip.y - wrist.y) > 0.12
        folded = all(lm[tid].y > lm[tid-2].y for tid in [8,12,16,20])
        return thumb_extended and thumb_down and folded

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            gesture_now = None
            direction_now = None

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                self.drawer.draw_landmarks(frame, res.multi_hand_landmarks[0],
                                           self.mp_hands.HAND_CONNECTIONS)

                # Detect quit gesture
                if self.is_thumbs_down(lm):
                    if self.quit_hold is None:
                        self.quit_hold = time.time()
                    elif time.time() - self.quit_hold >= 1.0:
                        gesture_now = "quit"
                else:
                    self.quit_hold = None
                    fingers = self.count_fingers(lm)
                    total = sum(fingers)
                    # Pause/play
                    if total <= 1:
                        gesture_now = "pause"
                    elif total >= 4:
                        gesture_now = "play"

                    # Direction based on index finger position
                    index_tip = lm[8]  # index finger tip
                    if index_tip.x < 0.4:
                        direction_now = "left"
                    elif index_tip.x > 0.6:
                        direction_now = "right"
                    elif index_tip.y < 0.4:
                        direction_now = "up"
                    elif index_tip.y > 0.6:
                        direction_now = "down"

            self.hist.append(gesture_now)
            cleaned = [x for x in self.hist if x]
            if cleaned:
                stable = max(set(cleaned), key=cleaned.count)
                with self.lock:
                    self.gesture = stable
                    self.direction = direction_now

            self.frame = frame

        self.cap.release()

# ===========================
# SNAKE GAME
# ===========================
pygame.init()
W, H = 600, 400
CELL = 20
win = pygame.display.set_mode((W, H))
pygame.display.set_caption("Snake - Index Finger Control")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

def place_food(snake):
    while True:
        x = random.randrange(0, W, CELL)
        y = random.randrange(0, H, CELL)
        if (x, y) not in snake:
            return (x, y)

def main():
    snake = [(100,100),(80,100),(60,100)]
    direction = (CELL, 0)
    food = place_food(snake)
    score = 0
    paused = False

    detector = GestureDetector()
    detector.start()

    last_move = time.time()
    speed = 0.12
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Handle gestures
        g = detector.get_gesture()
        if g == "pause":
            paused = True
        elif g == "play":
            paused = False
        elif g == "quit":
            running = False

        dir_g = detector.get_direction()
        if dir_g and not paused:
            if dir_g == "up" and direction != (0,CELL):
                direction = (0,-CELL)
            elif dir_g == "down" and direction != (0,-CELL):
                direction = (0,CELL)
            elif dir_g == "left" and direction != (CELL,0):
                direction = (-CELL,0)
            elif dir_g == "right" and direction != (-CELL,0):
                direction = (CELL,0)

        # Move snake
        if not paused and time.time() - last_move >= speed:
            last_move = time.time()
            x, y = snake[0]
            dx, dy = direction
            new = ((x+dx)%W, (y+dy)%H)

            if new in snake:
                snake = [(100,100),(80,100),(60,100)]
                direction = (CELL,0)
                paused = True
                score = 0
            else:
                snake.insert(0,new)
                if new == food:
                    score += 1
                    food = place_food(snake)
                else:
                    snake.pop()

        # Draw
        win.fill((10,10,10))
        pygame.draw.rect(win,(200,50,50),(food[0],food[1],CELL,CELL))
        for i,(x,y) in enumerate(snake):
            color = (50,200,50) if i==0 else (30,160,30)
            pygame.draw.rect(win,color,(x,y,CELL,CELL))
        hud = font.render(f"Score: {score} | Paused: {paused} | Gesture: {g} | Dir: {dir_g}", True,(220,220,220))
        win.blit(hud,(10,H-30))
        pygame.display.update()
        clock.tick(60)

        # Show camera
        if detector.frame is not None:
            cv2.imshow("Gesture Camera", detector.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

    detector.running = False
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
