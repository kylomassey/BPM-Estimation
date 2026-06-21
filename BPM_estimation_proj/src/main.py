from src.components.bpm_estimation import bpm_estimation
from src.components.note_detection.note_detection import chord_analyzer

def main():
    restart = True
    valid_choice = True
    while restart:
        filename = input("Please place the audio file in the music folder and enter the file name here: ")
        path = f"music/{filename}"
        while valid_choice:
            choice = input("For bpm estimation type \"bpm\". For chord analysis type \"chord\".: ")
            match choice.lower():
                case "bpm":
                    restart = bpm_estimation(path, filename)
                    valid_choice = False
                case "chord":
                    restart = chord_analyzer(path, filename)
                    valid_choice = False
                case _:
                    print("not an option")
        valid_choice = True

main()