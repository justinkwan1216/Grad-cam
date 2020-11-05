from object_detection import Object_detection
from algorithm import Algorithm
import time
a = Object_detection()
b = Algorithm()
while True:
    a.read_frame()
    a.get_indices()
    object_dict = a.extract_info_from_indices()
    
    # Algorithm
    b.feed_in(object_dict)
  
    # Call extract_and_update with the following parameters
    # (object_1_id, object_2_id,iou_threshold,trigger_alarm_threshold,
    # frame_buffer_threshold)
    
    # Max of trigger_alarm_threshold is frame_buffer_threshold
    # If trigger_alarm_threshold/frame_buffer_threshold is a alarm
    # alarm is triggered
        
    b.extract_and_update(0,2,0.1,44,50)
    
    # Print the real_alarm state for main object 0 with sub object 2
    #print(b.real_alarm["0_2"])
    
    a.show()

print(time.time()-now)
print(count)
