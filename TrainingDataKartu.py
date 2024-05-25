import ModulKlasifikasiCitraCNN as mCNN

# Nama folder datasetnya
DirektoriDataSet = "Kartu"

JumlahEpoh = 5

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

# Mulai training
mCNN.TrainingCNN(JumlahEpoh, DirektoriDataSet, LabelKelas,"BobotKartu.h5")