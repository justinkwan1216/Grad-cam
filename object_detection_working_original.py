import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause
import tensorflow as tf
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()
class Object_detection:
    def __init__(self, rtsp_link = "alarm-34010000001310000004-20200104_132427-8792-NoReflectiveVest.mp4"
                 ,ROI = [0,0,900,600], FPS = 10, classes = ["Person", "safety_shirt", "yellow_safety_shirt"],weight_file = "yolov3-custom_final.weights", config_file = "yolov3-custom.cfg"\
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
        self.ROI_Width = ROI[2]-ROI[0]
        self.ROI_Height = ROI[3]-ROI[1]
        self.FPS = FPS
        #--------------Initialization
        self.video = cv2.VideoCapture(self.rtsp_link)
        self.net = cv2.dnn.readNet(weight_file, config_file)
        #--------------Video info
        self.video_Width = self.video.get(3)
        self.video_Height =self.video.get(4)
        self.video_fps=self.video.get(5)
        #--------------Display info
        self.display_window_name = display_window_name
        self.f, self.axarr = plt.subplots(2,4)
        self.fig, self.ax = plt.subplots()
        #self.f.set_figheight(20)
        #self.f.set_figheight(20)
        #---------------Heatmap
    def read_frame(self):
        self.ret, self.image = self.video.read()
        self.image = self.image[int(self.ROI[1]):int(self.ROI[3]),int(self.ROI[0]):int(self.ROI[2])]
        self.blob = cv2.dnn.blobFromImage(self.image, self.scale, (self.input_width,self.input_height), (self.mean_sub_v_R,self.mean_sub_v_G,self.mean_sub_v_B), True, crop=False)
    def get_output_layers(self,net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    def get_conv_layers(self,layer,net):
        layer_names = net.getLayerNames()
        output_layers = layer
        return output_layers
    def get_relu_layers(self,layer,net):
        layer_names = net.getLayerNames()
        output_layers = layer
        return output_layers

    def stack_heatmap(self,num):
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img_rgb, alpha=.6)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        vis = np.zeros((320,320))
        for each in num:
            heatmap = cv2.resize(self.heatmap_array[each], (320, 320))
            heatmap = np.uint8(255 * heatmap)
            vis = np.sum((heatmap, vis),axis=0)
 
        self.ax.imshow(vis, cmap='jet', alpha=.4,extent=[xmin, xmax, ymin, ymax])
        #plt.show()
    def extract_grad_cam(self,num,num2,conv_layer,relu_layer,net,permu=False):
        conv_layer = self.net.forward(self.get_conv_layers(conv_layer,self.net))
        conv_layer = np.array(conv_layer)
        self.conv_layer = conv_layer.reshape((conv_layer.shape[2],conv_layer.shape[3],conv_layer.shape[4]))
        
        relu_layer = self.net.forward(self.get_relu_layers(relu_layer,self.net))
        relu_layer = np.array(relu_layer)
        if permu:
            self.relu_layer = relu_layer.reshape((relu_layer.shape[4],relu_layer.shape[3],relu_layer.shape[2]))
        else:
            self.relu_layer = relu_layer.reshape((relu_layer.shape[2],relu_layer.shape[3],relu_layer.shape[4]))
        multiply_product = np.multiply(self.conv_layer,self.relu_layer)
        #print(multiply_product.shape)
        sum_vec = np.sum(multiply_product, axis=0)
        heatmap = np.maximum(sum_vec, 0)
        heatmap /= np.max(heatmap)
        #print(sum_vec.shape)
        #plt.imshow(heatmap, cmap='jet')
        self.axarr[num,num2].imshow(heatmap, cmap='jet')
        draw(), pause(1e-3)
        self.heatmap_array.append(heatmap)

    def get_indices(self,conf_threshold=0.2,nms_threshold=0.4):
        self.net.setInput(self.blob)
        self.outs = self.net.forward(self.get_output_layers(self.net))
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.heatmap_array = []

        self.extract_grad_cam(0,0,['conv_0'],['relu_0'],self.net)
        self.extract_grad_cam(0,1,['conv_2'],['relu_2'],self.net)
        self.extract_grad_cam(0,2,['conv_4'],['relu_4'],self.net)
        self.extract_grad_cam(0,3,['conv_6'],['relu_6'],self.net)
        self.extract_grad_cam(1,0,['conv_8'],['relu_8'],self.net)
        self.extract_grad_cam(1,1,['conv_13'],['relu_13'],self.net)
        self.extract_grad_cam(1,2,['conv_10'],['permute_11'],self.net,True)
        self.extract_grad_cam(1,3,['conv_14'],['permute_15'],self.net,True)
        
        self.stack_heatmap([0,1,2,3,4,5])
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
