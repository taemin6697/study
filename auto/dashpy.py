#main.py  before while true
#import numpy as np
import time
from dashlib import dashlib
def dashmacSet():
    dash = dashlib.Dash()
    while True:
        try:
            dashmac=dash.find()
            break
        except:
            continue
    return dashmac
def dashConnect(dashmac):
    bot = dashlib.Command(dashmac)
    while True:
        try:
            bot.connect()
            break
        except:
            continue
    return bot
def dashReset():
    dash = dashlib.Dash()        
    while True:        
        try :
            dashmac = dash.find()
            print("Detected robot:{}".format(dashmac))
        except:
            continue

        bot = dashlib.Command(dashmac)
        try :
            bot.connect()                
            break
        except:
            continue
        
    return bot


#avg_x = (a * 2 + w) / 2
#avg_y = (b * 2 + h) / 2


#페이스 크기를 통한 거리재기 + 각도재기 ###각도/거리 = arctan() ###힘들면 거리 설정
#머리 대신 몸통 
def dashFaceCenter(bot, avg_x, center_x = 320, rl = 10):
    global range
    #bot = dashSet()
    if avg_x < center_x - range :
        bot.move(rl, 0, 1)
    elif avg_x > center_x + range :
        bot.move(0, rl, 1)
    else :
        bot.stop()
        dashCenter = False
        return dashCenter

def dashScanning(bot, Speed=10, during=1):
    bot.move(Speed,0, during)
    

def ComeOn(bot):
#     w = faces[0][2]
#     h = faces[0][3]
#     if w < face_size:   
    bot.move(20,20) #스피드 조정
#     else :
#         bot.stop()


def GoAway(bot):
    bot.move(36,0,1)
    bot.move(10,10)

def Spin(bot):
    bot.move(36,0)
def Left(bot):
    bot.move(0,20,1)
    time.sleep(1)
    bot.stop()
def Right(bot):
    bot.move(20,0,1)
    time.sleep(1)
    bot.stop()
def Stop(bot):
    bot.stop()

def Dash_Voice_Command(bot, i):
    #bot = dashSet()
    if i == 'ComeOn':
#         ComeOnStart = True
#         return ComeOnStart
        #로테이트 투 페이스
        ComeOn(bot)
    elif i == 'GoAway':
        GoAway(bot)
    elif i == 'Spin':
        Spin(bot)
    elif i == 'Left':
        Left(bot)
    elif i == 'Right':
        Right(bot)
    elif i == 'Stop':
        Stop(bot)



