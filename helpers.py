import cv2
import numpy as np
import time


def RGB_to_HSV(RGBColor:np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        RGBColor (tuple): (B,G,R) 

    Returns:
        tuple: (H,S,V) all in range [0,1]
    """
    
    B,G,R = RGBColor / 255
    Cmax = max(R,G,B)
    Cmin = min(R,G,B)
    deltaC = Cmax - Cmin
    H=0 # Hue
    
    if deltaC == 0:
        H=0
    elif Cmax == R:
        H = (1/6) * (((G-B)/deltaC) % 6)
        
    elif Cmax == G:
        
        H = (1/6) * (((B-R)/deltaC) + 2)
    elif Cmax == B:
        
        H = (1/6) * (((R-G)/deltaC) + 4)
        
    V = Cmax
    S = deltaC/Cmax if V > 0 else 0 # else is for V==0
    return np.array([H,S,V]) * 255

def get_image_names(name)->list[str]:
    # name = "dataset\\InstanceNames.txt"
    with open(name,"r") as f:
        text = f.read()
    name_list=text.splitlines()
    return name_list
def read_query(query_name,image_names):
    query = []
    for i in image_names:
        query += [read_image(query_name,i)]
    return np.array(query)
    
def read_image(folder_path:str = "dataset\\query_1",file_name ="Green_tailed_Towhee_0015_136678400.jpg") ->np.ndarray:
    """
    Returns:
        np.ndarray: Shape -> (96,96,3), BGR order
    """
    name = folder_path +"\\" + file_name
    image = cv2.imread(name)
    
    # cv2.imshow(file_name,image)
    # cv2.waitKey(0)
    return image

def check_is_hist_true(histogram:np.ndarray,origin,channelNum,quantNum):
    origin_hist,_ = np.histogram(origin[:,:,channelNum],bins = 16,range=(0,255))
    cmp = np.array(histogram[1] == origin_hist)
    cmp = np.array(histogram == origin_hist)
    if(cmp.all() == False):
        
        print("-----------------")
        print(f"DIFFERENT HISTOGRAM CHANNEL:{channelNum} QUANT: {quantNum}")
        print("My:")
        print(histogram[1])
        print("Np:")
        print(origin_hist)

        print("#################")
def check_is_3d_hist_true(histogram:np.ndarray,origin,quantNum):
    hist, edges = np.histogramdd(origin, bins=quantNum, range=((0, 256), (0, 256), (0, 256)))
    
    cmp = np.array(histogram == hist)
    if(cmp.all() == False):
        print("-----------------")
        print(f"DIFFERENT HISTOGRAM")
      
       
def l1_normalization(histogram:np.ndarray):
    return histogram / histogram.sum()


