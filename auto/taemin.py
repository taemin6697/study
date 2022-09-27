#It's Version2 code
from playsound import playsound
import os
import cv2
import speech_recognition as sr
import time
from gtts import gTTS
import tensorflow as tf
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np
from dashpy import dashReset, dashmacSet, dashConnect
from dashlib import dashlib
import natsort
from voicecmd import startVoiceCommand
from Button import PrepareButton,ButtonPushed
import shutil

X=[]
y=[]

def base_model(IMG_SHAPE):#기본 모델 불러오기 
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224,224,3),include_top=False,weights="imagenet"
        )
    model.trainable=False
    return model

def get_prediction_layer(n_classes):
    #여긴다시봐야함
    if n_classes == 2:
        prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        #_loss = tf.keras.losses.BinaryCrossentropy()
        _loss = "BinaryCrossentropy"
    else:
        prediction_layer = tf.keras.layers.Dense(n_classes, activation="softmax")
        #_loss = tf.keras.losses.sparse_categorical_crossentropy()
        _loss = "sparse_categorical_crossentropy"
    return prediction_layer,_loss

def get_model(IMG_SHAPE,prediction_layer,_loss):
    model = keras.Sequential()
    model.add(keras.Input(shape=(1, 1, 1024)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.5))
    model.add(prediction_layer)
    model.compile(loss=_loss,
              optimizer="rmsprop",
              metrics=["accuracy"])
    return model


def get_features_and_labels(base_model, dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.mobilenet_v3.preprocess_input(images)
        features = base_model.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def train(path,IMG_SIZE,n_classes):
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/start_train.mp3")
    train_dataset = image_dataset_from_directory(path,image_size=IMG_SIZE,batch_size=1)
    print('1')
    conv_model = base_model(IMG_SIZE)
    print('2')
    train_feature, train_labels = get_features_and_labels(conv_model,train_dataset)
    print('3')
    pre_layer,select_loss = get_prediction_layer(n_classes)
    print('4')
    model = get_model(IMG_SIZE,pre_layer,select_loss)
    print('5')
    model.fit(train_feature, train_labels,epochs=30)
    print('6')
    model.save('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_model.h5')
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/training...mp3")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # FP16 양자화 설정
    #converter.target_spec.supported_types = [tf.float16]
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # 모델 양자화
    tflite_model = converter.convert()
    # 변환된 모델을 .tflite 파일에 저장
    open("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/TFLITE_REAL.tflite", "wb").write(tflite_model)
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/say_training.mp3")
    #Moblienet 양자화
    converter_mobile = tf.lite.TFLiteConverter.from_keras_model(conv_model)
    tflite_model = converter_mobile.convert()
    open("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/TFLITE_MobileNet.tflite", "wb").write(tflite_model)
    print('train exist')
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/end_train.mp3")
    
def TFLITE_predict():

    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/start_predict.mp3")

    interpreter = tf.lite.Interpreter(model_path='/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/TFLITE_REAL.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter_mobile = tf.lite.Interpreter(model_path='/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/TFLITE_MobileNet.tflite')
    interpreter_mobile.allocate_tensors()
    input_details_mobile = interpreter_mobile.get_input_details()
    output_details_mobile = interpreter_mobile.get_output_details()

    

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    IMG_SIZE = (224,224)
    classes = search_folder()
    if len(classes) == 3:
        background_number = 0.9
    elif len(classes) == 4:
        background_number = 0.7
    elif len(classes) ==5:
        background_number = 0.5
    previous_class = classes[1]
    while True:
        but = 0
        
        PrepareButton(button)
        start = time.time()
        ret,frame = capture.read()
        frame_fliped = cv2.flip(frame, 1)
        #cv2.imshow("VideoFrame",frame_fliped)
        key = cv2.waitKey(50)
        if key > 4:
            break
        if(ButtonPushed(button[0])):
            but = 0
            break
        if(ButtonPushed(button[1])):
            but = 1
            break
        if(ButtonPushed(button[2])):
            but = 2
            break
        if(ButtonPushed(button[4])):
            but = 4
            break
        
        
        
        frame = cv2.resize(frame,IMG_SIZE)
        frame = np.array(frame)
        frame = frame.reshape(1, 224, 224, 3)
        #frame = keras.applications.mobilenet_v3.preprocess_input(frame)
        #mobilenet 추론
        #frame = conv_model.predict(frame)
        input_data = np.float32(frame)
        interpreter_mobile.set_tensor(input_details_mobile[0]["index"], input_data)
        interpreter_mobile.invoke()
        output_data = interpreter_mobile.get_tensor(output_details_mobile[0]["index"])
        output_data = np.squeeze(output_data)
        frame = output_data.reshape(1,1,1,1024)
        #real model 추론
        input_data = np.float32(frame)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        pre = np.squeeze(output_data)
 
        print(np.max(pre))
        real_name = ""

        if(np.max(pre)<0.7 and previous_class != 'background'):
            previous_class='background'
            print("background")
        elif(np.max(pre)>0.9 and previous_class != classes[np.argmax(pre)]):
            if('0' in classes[np.argmax(pre)]): 
                previous_class = classes[np.argmax(pre)]
                real_name = classes[np.argmax(pre)]
                real_name = real_name[:-1]
                print(classes[np.argmax(pre)]+'입니다.')
                
                tts = gTTS(text="{} 입니다.".format(real_name), lang="ko", slow=False)
                file_path = "/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm_real_name.mp3"
                if(os.path.exists(file_path)):
                    os.remove(file_path)
                tts.save(f"/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm_real_name.mp3")
                playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm_real_name.mp3")
            
            else:
                previous_class = classes[np.argmax(pre)]
                real_name = classes[np.argmax(pre)]
                print(classes[np.argmax(pre)]+'님 안녕하세요')
                tts = gTTS(text="{} 님 안녕하세요.".format(real_name), lang="ko", slow=False)
                file_path = "/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm_real_name.mp3"
                if(os.path.exists(file_path)):
                    os.remove(file_path)
                tts.save(f"/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm_real_name.mp3")
                playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm_real_name.mp3")
                
        print("predict_time :", time.time() - start)
    return but
def get_audio():#오디오를 받습니다.
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("wait...")
        #playsound("sound_effect/wait.mp3")
        time.sleep(0.5)
        print("recording...")
        #playsound("sound_effect/recording.mp3")
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language="ko-KR")

        except:
 
            print("Error")
            return ""

    return said


def get_name(i): #이름을 얻어 옵니다.
    i = int(i)
    while True:
        print("What's your name? (speak into the microphone)")
        playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/sound_effect/whatname.mp3")
        name = get_audio()
        if name == "":
            playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/ask_again.mp3")
            print("say again please...")
            continue
        else:
            print(name)
            tts = gTTS(text="{} 이 이름이 맞나요".format(name), lang="ko", slow=False)
            file_path = "/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm{i}.mp3"
            if(os.path.exists(file_path)):
                os.remove(file_path)
            tts.save(f"/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm{i}.mp3")
            print('저장은 됨')
            playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/confirm"+str(i)+".mp3")
            print("{} 이 이름이 맞나요?".format(name))
            #위를 대체할걸로
            #playsound("김태민 이 이름이 맞나요?로 해야함 ")
            response = get_audio()

            if response == "" or response in ["아니오", "아니요", "아니", "아리", "아니요 아니요"]:
                print("아니요")
                i+=1000
                continue

            else:
                print("네")
                break
    
    return name

def get_image(name,capture, i, IMG_SIZE):#이미지를 얻어오는 코드 입니다.
    count = 0
    camera_check = 0
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/camera_button.mp3")
    while True:
        _, frame = capture.read()
        # 이미지 뒤집기
        frame_fliped = cv2.flip(frame, 1)
        # 이미지 출력
        #cv2.imshow("VideoFrame", frame_fliped)

        key = cv2.waitKey(200)

        if camera_check >= 5:
            break
        if ButtonPushed(button[3]) == True:
            playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/sound_effect/capture.mp3")
            count += 1 
            frame = cv2.resize(frame, IMG_SIZE)
            X.append(frame)
            y.append(i)
            print('이미지 저장 실행')
            print(os.getcwd())
            if(check_set_folder(name)==True):
                saved_image(name,frame,i)
            else:
                saved_image(name,frame,random.randint(100,400))
            print('이미지 저장 실행 완료')
            i+=1
            camera_check+=1
    return count

def len_of_folder():
    search_dir=search_folder()
    return len(search_dir)

def check_set_folder(name):#폴더 중복 체크
    search_dir = search_folder()
    if name in search_dir:
        return False
    else:
        return True

def search_folder():#폴더 검색
    path2 = '/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/'
    count = 0
    for(path, dir, files) in os.walk(path2):
        search_dir = dir
        if count == 0:
            break
    folder_list = natsort.natsorted(search_dir)
    return folder_list

def creat_folder(name):#폴더를 생성합니다.
    search_dir = search_folder()
    if name in search_dir:
        print('파일이 이미 있습니다.')
    else:
        os.mkdir("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/"+name)

def saved_image(name,frame,i):#이미지를 생성합니다.
    i = str(i)
    #saved_tts/confirm{i}
    save_file = '/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/'+name+'/'+i+'.jpg'
    extension = os.path.splitext(save_file)[1]
    result,encoded_img = cv2.imencode(extension,frame)

    if result:
        with open(save_file,mode='w+b') as f:
            encoded_img.tofile(f)
    #print(save_file)
    #cv2.imwrite('data/'+name+'/'+i,frame)


def predict():
    model = load_model('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_model.h5')
    model.summary()
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    IMG_SIZE = (224,224)
    conv_model = base_model(IMG_SIZE)
    classes = search_folder()
    previous_class = classes[1]
    while True:
        start = time.time()
        ret,frame = capture.read()
        frame_fliped = cv2.flip(frame, 1)
        #cv2.imshow("VideoFrame",frame_fliped)
        key = cv2.waitKey(50)
        if key > 4:
            break
        frame = cv2.resize(frame,IMG_SIZE)
        frame = np.array(frame)
        frame = frame.reshape(1, 224, 224, 3)
        #frame = keras.applications.mobilenet_v3.preprocess_input(frame)
        frame = conv_model.predict(frame)
        print('프레임',type(frame),'모양',frame.shape)
        pre = model.predict(frame)
        print(np.max(pre))


        if(np.max(pre)<0.7 and previous_class != 'background'):
            previous_class='background'
            print("background")
        elif(np.max(pre)>0.8 and previous_class != classes[np.argmax(pre)]):
            if('0' in classes[np.argmax(pre)]): 
                previous_class = classes[np.argmax(pre)]
                print(classes[np.argmax(pre)]+'입니다.')
            else:
                previous_class = classes[np.argmax(pre)]
                print(classes[np.argmax(pre)]+'님 안녕하세요')
        print("time :", time.time() - start) 


def check_response(response):
    while True:
        response = get_audio()
        print(response)
        if response == "" or response not in [
            "Exit",
            "exit",
            "Stop",
            "stop",
            "No",
            "no",
            "Yes",
            "yes",
            "종료",
            "종로",
            "종료 종료",
            "아니오",
            "아니요",
            "아니",
            "아니요 아니요",
            "네",
            "예",
            "내",
        ]:
            print("say again please...")
            continue
        else:
            break
    return response


def Add_object(IMG_SIZE): #데이터를 추가하는 코드 입니다.
    global X,y
    classes = []
    i = 0
    negative_responses = ["아니오", "아니요", "아니", "아니요 아니요"]
    esc_responses = ["종료", "종로", "종료 종료", "Stop", "stop", "Exit", "exit"]
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/add_object.mp3")
    #카메라 객체 생성
    capture = cv2.VideoCapture(0)

    #카메라 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        name = get_name(i)
        name = name+'0'
        classes.append(name)
        creat_folder(name)
        count = get_image(name,capture,i,IMG_SIZE)     
        i+=1
        print('Do you want to proceed next class (yes,no,stop,exit and etc?)')
        playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/add_object_class.mp3")
        response = ""
        response = check_response(response)
        
        if response in esc_responses:
            print('exit')
            playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/add_end.mp3")
            break
        else:
            print('new object add')

def Add_data(IMG_SIZE): #데이터를 추가하는 코드 입니다.
    global X,y
    classes = []
    i = 0
    negative_responses = ["아니오", "아니요", "아니", "아니요 아니요"]
    esc_responses = ["종료", "종로", "종료 종료", "Stop", "stop", "Exit", "exit"]
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/add_person.mp3")
    #카메라 객체 생성
    capture = cv2.VideoCapture(0)
    
    #카메라 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        name = get_name(i)
        classes.append(name)
        creat_folder(name)
        count = get_image(name,capture,i,IMG_SIZE)     
        i+=1
        print('Do you want to proceed next class (yes,no,stop,exit and etc?)')
        playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/add_person_class.mp3")
        response = ""
        response = check_response(response)
        
        if response in esc_responses:
            print('exit')
            playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/add_end.mp3")
            break
        else:
            print('new person add')
            

def save_real_TFlite(path):
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open("model_TFlite.tflite", "wb").write(tflite_model)

def reset():
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/start_reset.mp3")
    shutil.rmtree('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/')
    os.mkdir('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data')
    

#Add_object(IMG_SIZE)
#Add_data(IMG_SIZE)
#train('data/',IMG_SIZE,su)
#print("완료!")
#path = 'data/retest/'
#model = load_model('saved_model.h5')
#model.summary()
#train_dataset = image_dataset_from_directory(path,image_size=IMG_SIZE,batch_size=1)
#conv_model = base_model(IMG_SIZE)
#test_feature, test_labels = get_features_and_labels(conv_model,train_dataset)
#score = model.evaluate(test_feature,test_labels,verbose=0)
#print(score)
#TFLITE_predict()
#get_audio()

#computer code*****************
#dashmac = dashmacSet()
#if dashmac is not None:
#    bot = dashConnect(dashmac)   
#print('connect') 
#if dashmac is None:
#    bot = dashReset()

#print("1. 데이터 추가")
#print("2. 사물 추가")
#print('3. 학습')
#print("4. 일반 추론")
#print("5. 양자화 추론")
#print('6.excute from sound')
#mode = int(input("모드를 골라주시오"))
#if mode == 1:
#    Add_data(IMG_SIZE)
#elif mode ==2:
#    Add_object(IMG_SIZE)
#elif mode==3:
#    train('data/',IMG_SIZE,su)
#elif mode==4:
#    predict()
#elif mode==5:
#    TFLITE_predict()
#elif mode==6:
#    startVoiceCommand(dashmac,bot)
#computer code*****************


#RPI CODE ***********

button = [13,36,31,29,15]
dashmac = dashmacSet()
if dashmac is not None:
    bot = dashConnect(dashmac)
IMG_SIZE = (224,224)
PrepareButton(button)
while(True):
    playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/start.mp3")
    su = len_of_folder()
    if(su<2):
        playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/tts_list/first_add.mp3")
        print('add Person ')
        Add_data(IMG_SIZE)
        print('add object')
        Add_object(IMG_SIZE)
        su = len_of_folder()
        train('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/',IMG_SIZE,su)
    #train('data/',IMG_SIZE,su)
    but = TFLITE_predict()
      
    while(True):
        print('but=',but)
        PrepareButton(button)
        if dashmac is None:
            bot = dashReset()
            
        if(but==0):
            print('add_person')
            Add_data(IMG_SIZE)
            su = len_of_folder()
            train('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/',IMG_SIZE,su)
            break
        if(but==1):
            print('add_object')
            Add_object(IMG_SIZE)
            su = len_of_folder()
            train('/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/data/',IMG_SIZE,su)
            break
        if(but==2):
            print('voice_command')
            playsound("/home/pi/Desktop/final_test/auto-TinyTeachableMachine_Final-TEST/saved_tts/voice_command.mp3")
            startVoiceCommand(dashmac,bot)
            break
        if(but==4):
            print('initalization')
            reset()
            break

    

