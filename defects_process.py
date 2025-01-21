from ultralytics import YOLO
from math import sqrt
import cv2
import numpy as np
import os
import shutil
def create_image(item,frame_Num):
    Draw_frame=item.plot()
    target_class=Script_param['model_for_DI'].names[int(item.boxes.cls.cpu())]
    target_folder=initial_param['output_folder']
    if (custom_param['folder_per_frame']):
        os.makedirs(f'{target_folder}/frame_{frame_Num}',exist_ok=True)
        cv2.imwrite(f'{target_folder}/frame_{frame_Num}/DI_{int(item.boxes.id.cpu())}_{target_class}.jpg', Draw_frame)
    else:
        cv2.imwrite(f'{target_folder}/frame_{frame_Num}_DI_{int(item.boxes.id.cpu())}_{target_class}.jpg', Draw_frame)
def results_analize(DI_info, Construct_info,current_frame_Num):
    if(have_correct_info(DI_info)):
        #Draw_frame=frame;
        for item in DI_info:
            if(Have_new_DI(item)):
                open_new_DI(item,Construct_info,current_frame_Num)
                create_image(item,current_frame_Num)
            else:
                calculate_DI(item,Construct_info)
            write_frame_with_DI(item,current_frame_Num)
def open_new_DI(DI,construction_info,start_frame_Num):
    item=DI.boxes
    template={'start':start_frame_Num,
              'QTY_frames':1,
              'at_construction':Detect_owner(DI,construction_info),
              'percent':float(item.conf.cpu().numpy())}
    cls=int(item.cls[0].cpu())
        #print(type(cls))
    id =int(item.id[0].cpu())
        #print(type(id))
        #Проверяем встретился ли нам новый дефект
    track_dict.get(cls)[id]=template
def near_box(DI,Construction_info):
    if(len(Construction_info)==0):
        return 'No_owner'
    else:
        x1=DI[0]
        y1=DI[1]
        w1=DI[2]
        h1=DI[3]
        distance=None
        for item in Construction_info:
            coord=item.boxes.xywh.cpu().tolist()[0]
            x2=coord[0]
            y2=coord[1]
            w2=coord[2]
            h2=coord[3]
            z=sqrt((abs(x1-x2)-(w1+w2)/2)**2+(abs(y1-y2)-(h1+h2)/2)**2)
            if z<0:
                return int(item.boxes.cls[0].cpu())
            if(distance==None or z<distance):
                distance=z
                result = item
        return int(result.boxes.cls[0].cpu())

def Detect_owner(DI,Construction_info):
    DI_coord= DI.boxes.xywh.cpu().tolist()[0]
    #print(DI_coord)
    if(custom_param['Construct_inbox_only']):
        for item in Construction_info:
            Construction_box = item.boxes.xyxy.cpu().numpy().astype(np.int32)[0]
            flag=inbox(DI_coord,Construction_box)
            if flag:
                return int(item.boxes.cls[0].cpu())
            else:
                return 'No_owner'
        return 'No_owner'
    else:
        return near_box(DI_coord,Construction_info)

def Have_new_DI(input):
    info=input.boxes
    cls=int(info.cls[0].cpu())
    id =int(info.id[0].cpu())
        #Проверяем встретился ли нам новый дефект
    if(track_dict.get(cls).setdefault(id)==None):
        return True
    else:
        return False

def have_correct_info(info_out):
    info=info_out.boxes
    if(len(info)!=0):
        #Проверяем способна ли нейронка отследить найденные объекты
        if(info.is_track == True):
            return True
        else:
            print('Информация не отслеживается')
    else:
        print('Нет информации')
    return False
def inbox(DI_Coord,Cons_coord):
    if ((DI_Coord[0]>Cons_coord[0]) and (DI_Coord[0]<Cons_coord[2]) and (DI_Coord[1]>Cons_coord[1]) and (DI_Coord[1]>Cons_coord[3])):
        return True
    else:
        return False
def calculate_DI(DI,Construction_info):

    item=DI.boxes
    cls=int(item.cls[0].cpu())
        #print(type(cls))
    id =int(item.id[0].cpu())
        #print(type(id))
        #Проверяем встретился ли нам новый дефект
    DI_indict=track_dict.get(cls).get(id)
    DI_indict['QTY_frames']=track_dict.get(cls).get(id).get('QTY_frames')+1
    #print(DI_indict['at_construction'])
    if(DI_indict['at_construction']=='No_owner'):
        #print('construction_fault')
        if(Detect_owner(DI,Construction_info)!='No_owner'):
            print('fixed')
            DI_indict['at_construction']=Detect_owner(DI,Construction_info)

def print_in_construct(i,j):
    temp = track_dict[i][j]['at_construction']
    if temp =='No_owner':
        return 'No info'
    else:
        return Script_param['model_for_construction'].names[temp]
def make_info_txt_file():
    rs=''
    for i in track_dict.keys():
        cls=Script_param['model_for_DI'].names[i]
        rs+=f'Class {cls}:\n'
        for j in track_dict[i].keys():
            if j=='frame_list':
                #print('skip')
                continue
            rs+=f'''\tDI №{j}
            Впервые замечен на кадре № {track_dict[i][j]['start']}({round(track_dict[i][j]['start']/Script_param['fps'],2)} секунда видео)
            Присутвует в кадре: {round(track_dict[i][j]['QTY_frames']/Script_param['fps'],2)} секунд
            Дефект на конструкции: {print_in_construct(i,j)}
            Вероятность дефекта: {round(track_dict[i][j]['percent']*100)}%\n\n'''
        rs+='\n\n'
        temp=initial_param['output_folder']
        file_name=f'{temp}/info.txt'
    with open(file_name, 'w') as f:
        f.write(rs)
def write_frame_with_DI(info,current_frame_Num):
    cls=int(info.boxes.cls[0].cpu())
    frame_list=track_dict.get(cls)['frame_list']
    if(current_frame_Num not in frame_list):
        track_dict.get(cls)['frame_list'].append(current_frame_Num)
def initiation():
    Script_param['capture']=cv2.VideoCapture(initial_param['video'])
    Script_param['model_for_DI']=YOLO(initial_param['DI'])
    Script_param['model_for_construction']=YOLO(initial_param['Construct'])
    Script_param['fps'] = int(Script_param['capture'].get(cv2.CAP_PROP_FPS))
    Script_param['QTY_FRAMES']=int(Script_param['capture'].get(cv2.CAP_PROP_FRAME_COUNT))
    for i in Script_param['model_for_DI'].names:
        track_dict[i]={'frame_list':[]}
    os.makedirs(initial_param['output_folder'],exist_ok=True)
def track(frame):
    if (custom_param['Deep_track']):
        return Script_param['model_for_DI'].track(frame,persist=True)[0]
    else:
        return Script_param['model_for_DI'].track(frame,persist=True,tracker="bytetrack.yaml")[0]
def start_analize():
    current_frame_Num=0
    while current_frame_Num<Script_param['QTY_FRAMES']:
        ret, frame = Script_param['capture'].read()
        current_frame_Num+=1
        if not ret:
            break
        if(custom_param['Resize_const']!=None):
            frame=cv2.resize(frame,custom_param['Resize_const'])
        DI_results = track(frame)
        Construct_results = Script_param['model_for_construction'](frame)[0]
        results_analize(DI_results, Construct_results,current_frame_Num)
        print(f'обработано {current_frame_Num}/{Script_param["QTY_FRAMES"]} кадров видео')
    Script_param['capture'].release()
def compile_all_info():
    make_info_txt_file()
    shutil.make_archive(initial_param['result_zip'], 'zip', initial_param['output_folder'])
Script_param={'capture':0,
              'fps':0,
              'QTY_FRAMES':0,
              'model_for_DI':0,
              'model_for_construction':0}
track_dict={}
#Настройки модели
                # путь к видео
initial_param={'video':'./input.mp4',
               # путь к модели для дефектов
              'DI':'DI_DETECTOR.pt',
               #путь к модели для конструкций
              'Construct':'Construct_DETECTOR.pt',
               #папка с изображениями и текстом
               'output_folder':'./output_data',
                #директория архива
               'result_zip':'./result'}


                #вариант отслеживания
custom_param={'Deep_track':False,
              #рассортировка кадров по папкам
              'folder_per_frame':True,
              #Жёстко ли определять пренадлежность к конструкциям
              'Construct_inbox_only':False,
              #Желаемый размер кадра
              'Resize_const':[640,480]}
