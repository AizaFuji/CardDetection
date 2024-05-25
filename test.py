import cv2
import numpy as np
import copy
import ModulKlasifikasiCitraCNN as mCNN
import random

########################## Inisiasi Kamera #######################
vid = cv2.VideoCapture(2)
# vid = 'D:/study/semester 5/pcv/TUGAS PCV/FP/contoh1.mp4'
# vid = cv2.VideoCapture(vid)
#Mengatur ukuran frame
fwidth = 640
fheight = 480

LabelKelas = (
    "Dua Club",
    "Tiga Club",
    "Empat Club",
    "Lima Club",
    "Enam Club",
    "Tujuh Club",
    "Delapan Club",
    "Sembilan Club",
    "Sepuluh Club",
    "Jack Club",
    "Queen Club",
    "King Club",
    "Ace Club",
    "Dua Diamond",
    "Tiga Diamond",
    "Empat Diamond",
    "Lima Diamond",
    "Enam Diamond",
    "Tujuh Diamond",
    "Delapan Diamond",
    "Sembilan Diamond",
    "Sepuluh Diamond",
    "Jack Diamond",
    "Queen Diamond",
    "King Diamond",
    "Ace Diamond",
    "Dua Heart",
    "Tiga Heart",
    "Empat Heart",
    "Lima Heart",
    "Enam Heart",
    "Tujuh Heart",
    "Delapan Heart",
    "Sembilan Heart",
    "Sepuluh Heart",
    "Jack Heart",
    "Queen Heart",
    "King Heart",
    "Ace Heart",
    "Dua Spade",
    "Tiga Spade",
    "Empat Spade",
    "Lima Spade",
    "Enam Spade",
    "Tujuh Spade",
    "Delapan Spade",
    "Sembilan Spade",
    "Sepuluh Spade",
    "Jack Spade",
    "Queen Spade",
    "King Spade",
    "Ace Spade",
)

#Load Model
model = mCNN.LoadModel("BobotKartu.h5")

#Fungsi untuk ngasih nilai tiap kartu
def getValue():
    value = {
        "Dua Club": 2,
        "Tiga Club": 3,
        "Empat Club": 4,
        "Lima Club": 5,
        "Enam Club": 6,
        "Tujuh Club": 7,
        "Delapan Club": 8,
        "Sembilan Club": 9,
        "Sepuluh Club": 10,
        "Jack Club": 11,
        "Queen Club": 12,
        "King Club": 13,
        "Ace Club": 14,
        "Dua Diamond": 2,
        "Tiga Diamond": 3,
        "Empat Diamond": 4,
        "Lima Diamond": 5,
        "Enam Diamond": 6,
        "Tujuh Diamond": 7,
        "Delapan Diamond": 8,
        "Sembilan Diamond": 9,
        "Sepuluh Diamond": 10,
        "Jack Diamond": 11,
        "Queen Diamond": 12,
        "King Diamond": 13,
        "Ace Diamond": 14,
        "Dua Heart": 2,
        "Tiga Heart": 3,
        "Empat Heart": 4,
        "Lima Heart": 5,
        "Enam Heart": 6,
        "Tujuh Heart": 7,
        "Delapan Heart": 8,
        "Sembilan Heart": 9,
        "Sepuluh Heart": 10,
        "Jack Heart": 11,
        "Queen Heart": 12,
        "King Heart": 13,
        "Ace Heart": 14,
        "Dua Spade": 2,
        "Tiga Spade": 3,
        "Empat Spade": 4,
        "Lima Spade": 5,
        "Enam Spade": 6,
        "Tujuh Spade": 7,
        "Delapan Spade": 8,
        "Sembilan Spade": 9,
        "Sepuluh Spade": 10,
        "Jack Spade": 11,
        "Queen Spade": 12,
        "King Spade": 13,
        "Ace Spade": 14
    }
    return value

def hitung_score(cards):
    value_dict = getValue()
    skor = sum(value_dict[card[0]] for card in cards)
    return skor
    
def DrawResizedCard(img, card_image, pos, size):
    resized_card = cv2.resize(card_image, size)
    img[pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0]] = resized_card
    return img

#Dictionary untuk gambar kartu
card_images = {}
for label in LabelKelas:
    card_images[label] = cv2.imread(f"card/{label}.jpg")

# Nambahin tulisan
def DrawText(img,sText,pos,color):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    posf        = pos
    fontScale   = .7
    fontColor   = color
    thickness   = 2
    lineType    = 2
    cv2.putText(img,sText,
        posf,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)
    return copy.deepcopy(img)


def preproses(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,50,50)
    kernel = np.ones((5,5))
    imgDilatasi = cv2.dilate(imgCanny,kernel,iterations=2)
    imgErosi = cv2.erode(imgDilatasi,kernel,iterations=1)

    return imgCanny

#Mendapatkan Kontur
def getcotours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>2000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    #cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 10)
    return biggest


#Membaca Video
success, frame = vid.read()

if not success:
    print("Gagal membaca video")
    exit()

# Inisialisasi game_frame hitam 
game_frame = np.zeros_like(frame)

# Inisialisasi list kartu
opened_card = []
player_card = []
computer_card = []

while True:
    success, frame = vid.read()

    if not success:
        break

    frame = cv2.resize(frame, (fwidth, fheight))
    imgContour = frame.copy()
    preprocessed_img = preproses(frame)
    biggest = getcotours(preprocessed_img)


    #Motong Bagian yang di kontur
    if len(biggest) != 0:
        x, y, w, h = cv2.boundingRect(biggest)
        cropped_frame = frame[y:y+h, x:x+w]
        cv2.imshow('Crop', cropped_frame)

        # Prepare image for prediction
        X = []
        image = cv2.resize(cropped_frame, (128, 128))
        image = np.asarray(image) / 255.0
        image = image.astype('float32')
        X.append(image)
        X = np.array(X)

        # Predict
        hs = model.predict(X, verbose=0)
        n = np.argmax(hs)  # Improved way to get index of maximum value

        label_text = LabelKelas[n]
        result_img = DrawText(frame.copy(), label_text, (200, 100),(0,0,255))  # Menambahkan argumen posisi teks
    else:
        result_img = frame.copy()

    cv2.imshow('Result Contour', imgContour)
    cv2.imshow('Result Prediction', result_img)

    key = cv2.waitKey(1) & 0xFF  # Define and get the pressed key
    if key == ord('z'):
        break
    # Tommbol spasi untuk menampilkan kartu pada game frame
    elif key == ord(' '):  # Check for spacebar press
        if len(biggest) != 0:
            
            opened_card.append((label_text, (0, 0, 255)))  
            player_card.append((label_text, (0, 0, 255)))  

            computer_take = random.choice(LabelKelas)
            opened_card.append((computer_take, (255, 0, 0)))  
            computer_card.append((computer_take, (255, 0, 0)))  

            game_frame = np.zeros_like(frame) 
            
            game_frame = DrawText(game_frame, "Computer Card", (15, int(fheight *0.05)), (255, 0, 0))
            game_frame = DrawText(game_frame, "Player Card", (15, int(fheight *0.5)), (0, 255, 0))
            
            # Menampilkan player card dan computer card
            card_size = (96, 144)  # Ubah ukuran sesuai kebutuhan
            for i, (text, _) in enumerate(opened_card):
                if i % 2 == 0:  # player
                    y_offset = int(fheight * 0.55)
                    x_offset = 20 + i * 50
                else:  # computer
                    y_offset = int(fheight * 0.1)
                    x_offset = 20 + (i-1) * 50

                card_img = card_images[text]
                game_frame = DrawResizedCard(game_frame, card_img, (x_offset, y_offset), card_size)

            cv2.imshow('GAME', game_frame)

    # Tombol q untuk menampilkan total skor 
    elif key == ord('q'): 
        #Mengitung total skor player dan komputer
        total_player = hitung_score(player_card)
        total_computer = hitung_score(computer_card)
        print(f"Total nilai kartu pemain: {total_player}")
        print(f"Total nilai kartu komputer: {total_computer}")
        # Menampilkan total nilai pada layar
        game_frame = DrawText(game_frame, f"Score Player: {total_player}", (350, int(fheight *0.5)), (0, 255, 0))
        game_frame = DrawText(game_frame, f"Score Computer: {total_computer}", (350, int(fheight *0.05)), (255, 0, 0) )
        cv2.imshow('GAME', game_frame)
    #Tombol a untuk menampilkan pemenang 
    elif key == ord('a'):
        # total_player = hitung_skor(player_card)
        # total_computer = hitung_skor(computer_card)
        winner = ""
        if total_player > total_computer:
            winner = "Player wins!"
        elif total_computer > total_player:
            winner = "Computer wins!"
        else:
            winner = "It's a tie!"

        print(winner)
        # Mengosongkan frame
        game_frame = np.zeros_like(frame)
        # Menampilkan pemenang pada konsol
        game_frame = DrawText(game_frame, winner, (int(fwidth * 0.45), int(fheight * 0.5)), (0, 0, 255))
        cv2.imshow('GAME', game_frame)



vid.release()
cv2.destroyAllWindows()
