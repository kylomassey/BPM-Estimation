from src.components.bpm_estimation import bpm_estimation
from src.components.note_detection.note_detection import stft_chord_analyzer, cqt_chord_analyzer

def main():
    restart = True
    valid_choice = True
    while restart:
        filename = input("Please place the audio file in the music folder and enter the file name here: ")
        path = f"music/{filename}"
        while valid_choice:
            choice = input("1 for bpm estimation, 2 for chord analysis: ")
            print(choice)
            match int(choice):
                case 1:
                    restart = bpm_estimation(path, filename)
                    valid_choice = False
                case 2:
                    choice = input("1 for CQT, 2 for STFT: ")
                    match int(choice):
                        case 1:
                            restart = cqt_chord_analyzer(path, filename)
                            valid_choice = False
                        case 2:
                            restart = stft_chord_analyzer(path, filename)
                            valid_choice = False
                        case _:
                            print("not an option")
                case _:
                    print("not an option")
        valid_choice = True

main()