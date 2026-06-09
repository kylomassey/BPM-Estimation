from src.components.bpm_estimation import bpm_estimation

def main():
    restart = True
    while restart:
        print("Please place the audio file in the music folder and enter the file name here:")
        filename = input()
        path = f"music/{filename}"
        restart = bpm_estimation(path, filename)

main()