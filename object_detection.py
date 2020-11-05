import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import figure, draw, pause
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()
class Object_detection:
    def __init__(self, rtsp_link = "video.mp4"
                 ,ROI = [0,0,1280,720], FPS = 10, classes = ["Person", "safety_shirt", "yellow_safety_shirt"],weight_file = "yolov3-custom_final.weights", config_file = "yolov3-custom.cfg"\
                 ,scale = 0.00392156862, input_height = 320, input_width = 320, mean_sub_v_R = 0, mean_sub_v_G = 0, mean_sub_v_B = 0, display_window_name = "object detection"):
        self.rtsp_link = rtsp_link
        self.ROI = ROI
        self.classes = classes
        weight_file = weight_file
        config_file = config_file
        self.scale = scale
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.input_height = input_height
        self.input_width = input_width
        self.mean_sub_v_R = mean_sub_v_R
        self.mean_sub_v_G = mean_sub_v_G
        self.mean_sub_v_B = mean_sub_v_B
        self.ROI_Width = self.ROI[2]-self.ROI[0]
        self.ROI_Height = self.ROI[3]-self.ROI[1]
        self.FPS = FPS
        #--------------Initialization
        self.video = cv2.VideoCapture(self.rtsp_link)
        self.net = cv2.dnn.readNet(weight_file, config_file)
        self.current_time = time.time()
        self.frame_count=0
        #--------------Video info
        self.video_Width = self.video.get(3)
        self.video_Height =self.video.get(4)
        self.video_fps=self.video.get(5)
        if self.ROI[2]>self.video_Width:
            self.ROI[2]=self.video_Width
        if self.ROI[0]<0:
            self.ROI[0]=0
        if self.ROI[3]>self.video_Height:
            self.ROI[3]=self.video_Height
        if self.ROI[0]<0:
            self.ROI[0]=0
        self.ROI_Width = self.ROI[2]-self.ROI[0]
        self.ROI_Height = self.ROI[3]-self.ROI[1]
        
        #--------------Display info
        self.display_window_name = display_window_name
        self.plot_arr_1=2
        self.plot_arr_2=5
        self.f, self.axarr = plt.subplots(self.plot_arr_1,self.plot_arr_2,figsize=(20, 8))
        self.fig, self.ax = plt.subplots()
        self.fig.show()
        self.f.show()
        #self.f.set_figheight(20)
        #self.f.set_figheight(20)
        #---------------Heatmap
        self.all_grad_cam_pair = ['conv_0','relu_0','conv_2','relu_2','conv_4','relu_4','conv_6','relu_6',\
                             'conv_8','relu_8','conv_13','relu_13','conv_10','permute_11','conv_14','permute_15','conv_10','yolo_11','conv_14','yolo_15']
        #self.all_grad_cam_pair = ['conv_0','relu_0','conv_2','relu_2','conv_4','relu_4','conv_6','relu_6',\
        #                     'conv_8','relu_8','conv_13','relu_13']
    def read_frame(self):
        if time.time()-self.current_time<1:
            if self.frame_count<10:
                read=True
            else:
                read=False
        else:
            read=True
            self.current_time = time.time()
            self.frame_count=0
        
        self.ret, self.image = self.video.read()
        if read:
            self.image = self.image[int(self.ROI[1]):int(self.ROI[3]),int(self.ROI[0]):int(self.ROI[2])]
            self.blob = cv2.dnn.blobFromImage(self.image, self.scale, (self.input_width,self.input_height), (self.mean_sub_v_R,self.mean_sub_v_G,self.mean_sub_v_B), True, crop=False)
    def get_output_layers(self,net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    def get_layers(self,layer,net):
        layer_names = net.getLayerNames()
        output_layers = layer
        print(layer_names)
        return output_layers

    def stack_heatmap(self,num):
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img_rgb, alpha=0.2)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        vis = np.zeros((self.input_height, self.input_width))
        for each in num:
            heatmap = cv2.resize(self.heatmap_array[each], (self.input_height, self.input_width))
            heatmap = np.uint8(255 * heatmap)
            vis = np.sum((heatmap, vis),axis=0)
 
        self.ax.imshow(vis, cmap='jet', alpha=.3,extent=[xmin, xmax, ymin, ymax])
        self.fig.savefig("result.png")
    def extract_grad_cam(self,all_grad_cam_pair,net,permu=False):
        a = self.net.forward(self.get_layers(['yolo_11'],self.net))
        a = np.array(a)
        
        for each in a:    
            for j in each:
                if j[5]>0 or j[6]>0 or j[7]>0:
                    print(j)
        all_grad_cam_pair_layer = self.net.forward(self.get_layers(all_grad_cam_pair,self.net))
        for i in range(len(all_grad_cam_pair_layer)):
            current = all_grad_cam_pair_layer[i]
            current = np.array(current)
            #print(current.shape)
            if i%2==0:
                conv_layer = current.reshape((current.shape[1],current.shape[2],current.shape[3]))
            else:
                if "permute" in all_grad_cam_pair[i]:
                    act_layer = current.reshape((conv_layer.shape[0],conv_layer.shape[1],conv_layer.shape[2]))
                else:
                    act_layer = current.reshape((conv_layer.shape[0],conv_layer.shape[1],conv_layer.shape[2]))
                    
                

                grad = np.gradient((conv_layer, act_layer))
                sum_grad = np.sum(grad,axis=(0,1,2))

                heatmap = np.maximum(sum_grad, 0)

                
                heatmap /= np.max(heatmap)
                

                index = int((i-1)/2)
                self.axarr[int(index/self.plot_arr_2),index%self.plot_arr_2].imshow(heatmap, cmap='jet')
                self.heatmap_array.append(heatmap)
        self.f.savefig("heatmap.png")

    def get_indices(self,conf_threshold=0.2,nms_threshold=0.4):
        self.net.setInput(self.blob)
        self.outs = self.net.forward(self.get_output_layers(self.net))
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.heatmap_array = []
        
        self.extract_grad_cam(self.all_grad_cam_pair,self.net)

        self.stack_heatmap([0,1,2,3,4,5])
        
        self.f.canvas.draw()
        self.fig.canvas.draw()
        #draw()
        #pause(1e-5)

        
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * self.ROI_Width)
                    center_y = int(detection[1] * self.ROI_Height)
                    w = int(detection[2] * self.ROI_Width)
                    h = int(detection[3] * self.ROI_Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    self.class_ids.append(class_id)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])

        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, conf_threshold, nms_threshold)

    def extract_info_from_indices(self):
        self.info_array=[]
        self.object_dict={}
        for i in self.indices:
            i = i[0]
            box = self.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.object_dict.setdefault(self.class_ids[i],[]).append((self.confidences[i],[x,y,x+w,y+h]))
            self.info_array.append((self.class_ids[i],self.confidences[i],[x,y,x+w,y+h]))
            self.draw_prediction(self.image, self.class_ids[i], self.confidences[i], round(x), round(y), round(x+w), round(y+h))
        return self.object_dict
        # self.info_array is a 2d array contains all the info element, in format [(object_id, confidence level, [x1, y1, x2, y2]), (object_id, confidence level, [x1, y1, x2, y2])]
#--------------------------Draw to frame-------------------------------"
    def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    	
#---------------------------Display---------------------------------"
    def show(self):
        cv2.imshow(self.display_window_name, self.image)
        cv2.waitKey(1) == ord('q')
