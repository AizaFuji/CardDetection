import cv2
import numpy as np
import random
import copy
import ModulKlasifikasiCitraCNN as mCNN


# ==============================
#  LOAD RESOURCES
# ==============================
def load_resources():
    LabelKelas = tuple([
        f"{angka} {jenis}"
        for jenis in ["Club", "Diamond", "Heart", "Spade"]
        for angka in ["Dua", "Tiga", "Empat", "Lima", "Enam", "Tujuh",
                      "Delapan", "Sembilan", "Sepuluh",
                      "Jack", "Queen", "King", "Ace"]
    ])

    # Load model
    model = mCNN.LoadModel("BobotKartu.h5")

    # Value mapping
    value_map = {
        label: (
            2 + ["Dua","Tiga","Empat","Lima","Enam","Tujuh","Delapan","Sembilan","Sepuluh",
                 "Jack","Queen","King","Ace"].index(label.split()[0])
        )
        for label in LabelKelas
    }

    # Load card images
    card_images = {label: cv2.imread(f"card/{label}.jpg") for label in LabelKelas}

    return LabelKelas, model, card_images, value_map


# ==============================
#  PREPROCESSING + CONTOUR
# ==============================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, 50, 50)
    return canny


def get_biggest_contour(img, contour_frame):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                max_area = area
                biggest = approx
                cv2.drawContours(contour_frame, [approx], -1, (255, 0, 0), 3)

    return biggest


# ==============================
#  PREDIKSI KARTU
# ==============================
def detect_card(frame, model, LabelKelas, imgContour):
    processed = preprocess(frame)
    biggest = get_biggest_contour(processed, imgContour)

    if biggest is None:
        return None, imgContour

    x, y, w, h = cv2.boundingRect(biggest)
    crop = frame[y:y+h, x:x+w]

    # Resize + Predict
    img = cv2.resize(crop, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    label = LabelKelas[np.argmax(pred)]

    return label, imgContour


# ==============================
#  DRAW UTILITIES
# ==============================
def draw_text(img, text, pos, color):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return img


def draw_card(img, card_img, pos, size):
    resized = cv2.resize(card_img, size)
    x, y = pos
    img[y:y+size[1], x:x+size[0]] = resized
    return img


# ==============================
#  GAME LOGIC
# ==============================
def update_score(player_card, computer_card, value_map, score_player, score_computer):
    if not player_card or not computer_card:
        return score_player, score_computer

    p = value_map[player_card[-1]]
    c = value_map[computer_card[-1]]

    if p > c:
        score_player += 1
    elif c > p:
        score_computer += 1

    return score_player, score_computer


def draw_game_frame(frame_shape, opened_card, card_images, score_p, score_c):
    H, W, _ = frame_shape
    game_frame = np.zeros((H, W, 3), dtype=np.uint8)

    draw_text(game_frame, "Computer Card", (15, int(H*0.05)), (255, 0, 0))
    draw_text(game_frame, "Player Card", (15, int(H*0.5)), (0, 255, 0))

    size = (96, 144)

    for i, (label, owner) in enumerate(opened_card):
        y = int(H*0.55) if owner == "player" else int(H*0.1)
        x = 20 + i * 50
        game_frame = draw_card(game_frame, card_images[label], (x, y), size)

    draw_text(game_frame, f"Score Player: {score_p}", (350, int(H*0.5)), (0, 255, 0))
    draw_text(game_frame, f"Score Computer: {score_c}", (350, int(H*0.05)), (255, 0, 0))

    return game_frame


# ==============================
#  MAIN
# ==============================
def main():
    vid = cv2.VideoCapture(2)

    LabelKelas, model, card_images, value_map = load_resources()

    opened_card = []
    player_card = []
    computer_card = []
    score_p = 0
    score_c = 0

    while True:
        success, frame = vid.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        imgContour = frame.copy()

        label, imgContour = detect_card(frame, model, LabelKelas, imgContour)

        if label:
            result = draw_text(frame.copy(), label, (200, 100), (0, 0, 255))
        else:
            result = frame.copy()

        cv2.imshow("Contour", imgContour)
        cv2.imshow("Prediction", result)

        key = cv2.waitKey(1) & 0xFF

        # EXIT
        if key == ord('z'):
            break

        # OPEN CARD
        if key == ord(' ') and label:
            player_card.append(label)
            computer_pick = random.choice(LabelKelas)
            computer_card.append(computer_pick)

            opened_card.append((label, "player"))
            opened_card.append((computer_pick, "computer"))

            score_p, score_c = update_score(player_card, computer_card, value_map, score_p, score_c)

            game = draw_game_frame(frame.shape, opened_card, card_images, score_p, score_c)
            cv2.imshow("GAME", game)

        # SHOW WINNER
        if key == ord('a'):
            winner = "Player Wins!" if score_p > score_c else (
                "Computer Wins!" if score_c > score_p else "Draw!"
            )
            final = np.zeros_like(frame)
            draw_text(final, winner, (200, 200), (0,0,255))
            cv2.imshow("GAME", final)

    vid.release()
    cv2.destroyAllWindows()


main()
