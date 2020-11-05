class Algorithm:
    def __init__(self):
        #--------------Array to save info
        self.frame_state_array={}
        self.frame_state = {}
        self.real_alarm = {}
    def feed_in(self, object_dict):
        self.object_dict = object_dict
        # self.object_dict is dictionary that contain key-value pairs, that key is the object id, value_1 is the confidence level and value_2 is the bbox coordinates.
        # Dictionary in format {key: [(confidence_level, [x1, y1, x2, y2]), (confidence_level, [x1, y1, x2, y2])]}
        #for each in (info_array):
        #    self.object_dict.setdefault(each[0],[]).append((each[1],each[2]))
            
        # In self.object_dict[x][y][z], x indicates the object_id, y indicates the nth object in object_id x
        # and z indicates the info (0 for condifence level and 1 for coordinates for the bbox)    
        


    def bb_intersection_over_union(self,boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
 
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
 
        return iou
    
    # Apply IOU to object1 with object2, object1 is the main object to check and object2 is the sub
    def apply_iou(self, object_1_id, object_2_id,iou_threshold):
        Low_IOU_array=[]
        # Get the corresponds object info for the two objects
        object_1 = self.object_dict[object_1_id] if object_1_id in self.object_dict else []
        object_2 = self.object_dict[object_2_id] if object_2_id in self.object_dict else []
        
        for obj1 in object_1:
            Low_IOU_array.append(obj1)
            for obj2 in object_2:
                # Check the IOU percentage of (object 1 and object 2)'s bounding box
                if (self.bb_intersection_over_union(obj1[1],obj2[1]) > iou_threshold):
                    del Low_IOU_array[-1]
                    break
        return Low_IOU_array
    
    # Get alarm state bu calling apply_iou, return True if there is no object with low IOU
    def generate_alarm_state(self, object_1_id, object_2_id,iou_threshold):
        # Get the Low_IOU_array by calling apply_iou
        Low_IOU_array = self.apply_iou(object_1_id, object_2_id,iou_threshold)
        if len(Low_IOU_array)==0:
            return True
        else:
            return False
        
    # Store all alarm states into the frame_state_array by calling generate_alarm_state, store up to <frame_buffer_threshold> of frames
    def store_frame_state(self, object_1_id, object_2_id,name,iou_threshold,frame_buffer_threshold):
        # Get the alarm_state from the current frame with iou_threshold
        frame_state = self.generate_alarm_state(object_1_id, object_2_id,iou_threshold)
        
        self.frame_state_array.setdefault(name,[]).append(frame_state)
        self.frame_state_array[name]=self.frame_state_array[name][-frame_buffer_threshold:]
        #print(self.frame_state_array[name])

    # Update the info of frame_state_array by calling store_frame_state, and extract the info and update the corresponds keys in the real_alarm[<name>] dictionary to a state (True or False)                                                                           
    def extract_and_update(self, object_1_id, object_2_id,iou_threshold,trigger_alarm_threshold,frame_buffer_threshold):
        name = str(object_1_id)+"_"+str(object_2_id)
        
        self.store_frame_state(object_1_id, object_2_id,name,iou_threshold,frame_buffer_threshold)
        if self.frame_state_array[name].count(False)<trigger_alarm_threshold:
            self.real_alarm[name] = True
        else:
    	    self.real_alarm[name] = False
