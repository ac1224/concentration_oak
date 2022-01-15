"""
Illegal instruction (コアダンプ)
のエラーが出た場合には
export OPENBLAS_CORETYPE=ARMV8
を行う
"""


import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai # depthai - access the camera and its data packets
import blobconverter # blobconverter - compile and download MyriadX neural network

from scipy.spatial.distance import euclidean

import cv2
import time
from datetime import datetime, timedelta
import pickle

from imutils.video import FPS
fps = FPS()

from tools import *

import requests


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 1, self.line_type)
    def rectangle(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 1)

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)



# 1.7.7 Define a pipeline
#piplineはOAK内部のもの
pipeline = depthai.Pipeline()

# メモ
#サイズの定義
w = 300
h = 300



cam = pipeline.create(depthai.node.ColorCamera)
cam.setPreviewSize(w, h)
cam.setInterleaved(False)

#画像の出力
cam_to_jetson = pipeline.createXLinkOut()
cam_to_jetson.setStreamName("cam_out")
cam.preview.link(cam_to_jetson.input)



model_nn1 = pipeline.createNeuralNetwork()
model_nn1.setBlobPath("models/face-detection-retail-0004_openvino_2020_1_4shave.blob")
# cam_rgb.preview.link(model_nn1.input)
model_nn2 = pipeline.createNeuralNetwork()
model_nn2.setBlobPath("models/face_landmark_160x160_openvino_2020_1_4shave.blob")


model_in2 = pipeline.createXLinkIn()
model_in2.setStreamName("land68_in")


model_nn1_to_jetson = pipeline.createXLinkOut()
model_nn1_to_jetson.setStreamName("face_nn")
model_nn_to_jetson2 = pipeline.createXLinkOut()
model_nn_to_jetson2.setStreamName("land68_nn")

cam.preview.link(model_nn1.input)
model_nn1.out.link(model_nn1_to_jetson.input)

model_in2.out.link(model_nn2.input)
model_nn2.out.link(model_nn_to_jetson2.input)





# パイプラインが定義されたので、パイプラインでデバイスを初期化して起動します。


# 1.7.8 Initialize the DepthAI Device
with depthai.Device(pipeline) as device:



    cam_out = device.getOutputQueue("cam_out", 4, False)
    face_nn = device.getOutputQueue("face_nn",4,False)
    land68_in = device.getInputQueue("land68_in",4,False)
    land68_nn = device.getOutputQueue("land68_nn",4,False)
    # これらは結果でいっぱいになるので、次にやるべきことは結果を消費することです。
    # 2つのプレースホルダーが必要です。1つはrgbフレーム用、もう1つはnnの結果用です。
    frame = None
    detections = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]

        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    

    # 1.7.10 Consuming the results

    #　関数設定
    def run_nn(x_in, x_out, in_dict):
        nn_data = depthai.NNData()
        for key in in_dict:
            nn_data.setLayer(key, in_dict[key])
        x_in.send(nn_data)

        def wait_for_results(queue):
            start = datetime.now()
            while not queue.has():
                if datetime.now() - start > timedelta(seconds=1):
                    return False
            return True
        
        has_results = wait_for_results(x_out)
        if not has_results:
            raise RuntimeError("No data from nn!")
        return x_out.get()

    def eye_aspect_ratio(eye):
        A = euclidean(eye[1],eye[5])
        B = euclidean(eye[2],eye[4])
        C = euclidean(eye[0],eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(mouth):
        A = np.linalg.norm(mouth[1] - mouth[5])  # 51, 59
        B = np.linalg.norm(mouth[2] - mouth[4])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[3])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def send_result(result_dict):
        """
        data send to google
        """
        response = requests.post('https://script.google.com/macros/s/AKfycbzfXr0x9V9RsE7bqp5fD7dG0xKcLT3dQhAqURYKeVqlcILvvDh6/exec', 
                data = result_dict)
    #初期値設定

    period = 60 # 60secごとにデータ集計
    before_timing = -1 
    start_time_unix = time.time()
    tmp_data=np.empty((0,3))
    all_data=[]
    while True:
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration()


        current_time_unix = time.time()
        # 定期的なデータ集計＆保存
        d = round(current_time_unix - start_time_unix)
        
        if (d % period == 0) and (d!=0)  and (tmp_data.shape[0] >= 2) \
            and (d // period != before_timing):

            before_timing = d // period
            # print("before_timing:",before_timing)
            current_time = datetime.now()
            # print(tmp_data.shape)
            tmp_data_sum = np.sum(tmp_data, axis=0)

            #単位時間あたりのデータの集計
            face_ratio =  tmp_data_sum[1]/tmp_data.shape[0]
            gaze_ratio =  tmp_data_sum[2]/tmp_data.shape[0]

            # 顔認識がされない場合
            if face_ratio != 0:
                cnst_ratio = gaze_ratio / face_ratio
            else:
                cnst_ratio = 0
                
            print([current_time, face_ratio, gaze_ratio, cnst_ratio])
            all_data.append([current_time, face_ratio, gaze_ratio, cnst_ratio])
            # save list
            f = open('all_data.txt', 'wb')
            pickle.dump(all_data, f)

            # data send to gooogle
            result_dict = {}
            result_dict["time"] = str(current_time)
            result_dict["face_recon"] = str(face_ratio)
            result_dict["gaze_check"] = str(gaze_ratio)
            result_dict["gaze_ratio"] = str(cnst_ratio)
            send_result(result_dict)
            
            # reest
            tmp_data=np.empty((0,3))
   

        # カメラのフレームを確保する
        in_rgb = cam_out.tryGet()
        if in_rgb is not None:
            frame = np.array(in_rgb.getData()).reshape((3, 
                                        in_rgb.getHeight(), 
                                        in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8
                                        )

            # cv2.imshow("preview", frame)
            copied_frame = frame.copy()


            #顔の分析結果を取得する
            nn_data = face_nn.tryGet()
        else:
            tmp_data = np.vstack((tmp_data, np.array([current_time_unix,0,0])))
        
        if in_rgb is not None and nn_data is not None:
            try:
                arr = np.array(nn_data.getFirstLayerFp16())
                arr = arr[:np.where(arr == -1)[0][0]]
                results = arr.reshape((arr.size // 7, 7))

                face_coords = [
                    frame_norm(frame, *obj[3:7])
                    for obj in results
                    if obj[2] > 0.4
                ]

                if len(face_coords) == 0:
                    run_face  =  False

                if len(face_coords) > 0:
                    for face_coord in face_coords:
                        face_coord[0] -= 15
                        face_coord[1] -= 15
                        face_coord[2] += 15
                        face_coord[3] += 15
                    face_frame = [frame[
                        face_coord[1]:face_coord[3],
                        face_coord[0]:face_coord[2]
                    ] for face_coord in face_coords]

                    run_face  = True
                
                face_success = run_face 


                if face_success:
                    # print('Having seat')
                    for i in range(len(face_frame)):

                        # def run_land68(self,face_frame,count):
                        try:
                            count = i
                            face_frame =  face_frame[i]         
                            shape = (160,160)
                            data = [val for channel in cv2.resize(face_frame, shape).transpose(2, 0, 1) \
                                    for y_col in channel for val in y_col]
                            
                            nn_data = run_nn(land68_in,
                                            land68_nn, 
                                            {"data": data})
                            
                            out = np.array(nn_data.getFirstLayerFp16())
                            result = frame_norm(face_frame,*out)
                            eye_left = []
                            eye_right = []
                            mouth = []
                            hand_points = []
                            for i in range(72,84,2):
                                eye_left.append((out[i],out[i+1]))
                            for i in range(84,96,2):
                                eye_right.append((out[i],out[i+1]))
                            for i in range(96,len(result),2):
                                if i == 100 or i == 116 or i == 104 or i == 112 or i == 96 or i == 108:
                                    mouth.append(np.array([out[i],out[i+1]]))

                            for i in range(16,110,2):
                                if i == 16 or i == 60 or i == 72 or i == 90 or i == 96 or i == 108:
                                    pass
                                    cv2.circle(copied_frame,
                                            (result[i]+face_coords[count][0],result[i+1]+face_coords[count][1]),
                                            2,
                                            (255,0,0),
                                            thickness=1,
                                            lineType=8,
                                            shift=0)
                                    hand_points.append((result[i]+face_coords[count][0],result[i+1]+face_coords[count][1]))
                            

                            ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = \
                                get_pose_estimation(frame.shape, np.array(hand_points,dtype='double'))

                            ret, pitch, yaw, roll = get_euler_angle(rotation_vector)

                            # cv2.projectPoints顔の座標系とカメラの座標系の関係を tvecs, rvecs に指定し
                            # 入力点群が自動的にカメラ座標系に変換され，そののちに画像に投影する．
                            (nose_end_point2D, jacobian) = cv2.projectPoints(
                                                            np.array([(0.0, 0.0, 1000.0)]), 
                                                            rotation_vector, 
                                                            translation_vector, 
                                                            camera_matrix, 
                                                            dist_coeffs)

                            #p1が鼻側、p2が顔の方向
                            p1 = ( int(hand_points[1][0]), int(hand_points[1][1]))
                            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                            cv2.line(copied_frame, p1, p2, (255,0,0), 2)
                            
                            # calc angle
                            p1_arr = np.array(p1)
                            p2_arr = np.array(p2)
                            vec = p1_arr - p2_arr
                            angle = np.arctan2(vec[0], vec[1])
                            cv2.putText(copied_frame,
                                        text = str(angle), 
                                        org = (30, 20),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, 
                                        color=(0, 0, 255), 
                                        thickness=1)

                            # このangleの範囲設定がキモ
                            if (-1.5 > angle) or (angle > 1.6):
                                """
                                視線がディスプレイに向いてないと思われる時
                                """
                                tmp_data = np.vstack((tmp_data, np.array([current_time_unix,1,0])))
                            else:
                                """
                                視線がディスプレイに向いていると思われる時
                                """
                                tmp_data = np.vstack((tmp_data, np.array([current_time_unix,1,1])))

                            
                            left_ear = eye_aspect_ratio(eye_left)
                            right_ear = eye_aspect_ratio(eye_right)
                            ear = (left_ear + right_ear) / 2.0
                        
                            
                        except:
                            pass
                    
                        fps.update()
                else:
                    """
                    顔が認識されていない時
                    """
                    tmp_data = np.vstack((tmp_data, np.array([current_time_unix,0,0])))
                
            
            
            except:
                pass
            aspect_ratio = frame.shape[1] / frame.shape[0]
            cv2.imshow("Camera_view", cv2.resize(copied_frame, ( int(300),  int(300 / aspect_ratio))))





