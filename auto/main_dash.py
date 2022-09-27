
from dashlib import dashlib
from voice import bot_audio, add_voice_command, delete_voice_command, excute_voice_command

def main():
    dash = dashlib.Dash()
    dashmac = dash.find()
    print("Detected robot:{}".format(dashmac))
    if dashmac is None:
        exit(0)
        
    bot = dashlib.Command(dashmac)
    bot.connect()

    mode = int(input("mode: "))
    if mode == 4:
        while True:
            audio = bot_audio(1)
            excute_voice_command(audio, bot)






if __name__ == "__main__":
    main()
