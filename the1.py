from helpers import *

start = time.time()

image_names = get_image_names(name="dataset\\InstanceNames.txt")
NUM_OF_IMG= len(image_names)    
MAX_VAL = 255 
PER_CHANNEL = "per_channel"
THREE_D = "3d"
RGB= "RGB"
HSV= "HSV"
query1 = read_query("dataset\\query_1",image_names)
query2 = read_query("dataset\\query_2",image_names)
query3 = read_query("dataset\\query_3",image_names)
supports = read_query("dataset\\support_96",image_names)

query1_hsv = np.zeros(query1.shape)
query2_hsv = np.zeros(query2.shape)
query3_hsv = np.zeros(query3.shape)
supports_hsv = np.zeros(supports.shape)

def per_channel_color_histogram(query:np.ndarray,interval:int):
    query_hist = []
    for image in query:
        quantized = image//interval
        bin_number = (MAX_VAL+1)//interval 
        #For Blue C=0
        blue = quantized[:,:,0]
        unique, counts = np.unique(blue, return_counts=True)
        blueHistogram = np.array([unique,counts])

        #For Green C=1
        green = quantized[:,:,1]
        unique, counts = np.unique(green, return_counts=True)
        greenHistogram = np.array([unique,counts])
        
        #For Red C=2
        red = quantized[:,:,2]
        unique, counts = np.unique(red, return_counts=True)
        redHistogram = np.array([unique,counts])
        
        for i in range(bin_number):
            if i not in blueHistogram[0]:
                blueHistogram = np.insert(blueHistogram,i,[i,0],axis=1)
            if i not in greenHistogram[0]:
                greenHistogram = np.insert(greenHistogram,i,[i,0],axis=1)
            if i not in redHistogram[0]:
                redHistogram = np.insert(redHistogram,i,[i,0],axis=1)
        query_hist += [[blueHistogram[1],greenHistogram[1],redHistogram[1]]]
    return np.array(query_hist)

def per_channel_run(interval,rgb_hsv): 
    if rgb_hsv == RGB:
        support_hist = per_channel_color_histogram(supports,interval)
        query1_hist = per_channel_color_histogram(query1,interval)
        query2_hist = per_channel_color_histogram(query2,interval)
        query3_hist = per_channel_color_histogram(query3,interval)
        
    if rgb_hsv == HSV:
        support_hist = per_channel_color_histogram(supports_hsv,interval)
        query1_hist = per_channel_color_histogram(query1_hsv,interval)
        query2_hist = per_channel_color_histogram(query2_hsv,interval)
        query3_hist = per_channel_color_histogram(query3_hsv,interval)
        
    print_accuracies(support_hist,query1_hist,query2_hist,query3_hist,interval,rgb_hsv,PER_CHANNEL)       
    
def threeD_color_histogram(query:np.ndarray,interval:int):
    query_hist = []
    for image in query:
        quantized = image // interval
        bin_number = (MAX_VAL+1)//interval 
        
        blue = quantized[:,:,0]
        green = quantized[:,:,1]
        red = quantized[:,:,2]
        histogram = np.zeros((bin_number,bin_number,bin_number))
        h,w,_ = image.shape
        for i in range(h):
            for j in range(w):
                histogram[int(blue[i,j]), int(green[i,j]),int (red[i,j])] += 1
        query_hist += [[histogram]]
    return np.array(query_hist)
 
def threeD_run(interval,rgb_hsv:str):
    support_hist = []
    if rgb_hsv == RGB:
        support_hist = threeD_color_histogram(supports,interval)
        query1_hist = threeD_color_histogram(query1,interval)
        query2_hist = threeD_color_histogram(query2,interval)
        query3_hist = threeD_color_histogram(query3,interval)

    elif rgb_hsv == HSV:
        support_hist = threeD_color_histogram(supports_hsv,interval)
        query1_hist = threeD_color_histogram(query1_hsv,interval)
        query2_hist = threeD_color_histogram(query2_hsv,interval)
        query3_hist = threeD_color_histogram(query3_hsv,interval)
    print_accuracies(support_hist,query1_hist,query2_hist,query3_hist,interval,rgb_hsv,THREE_D)           
    

def compute_similarity(hists1:np.ndarray,hists2:np.ndarray):
    
    normalizedHist1 = l1_normalization(hists1)
    normalizedHist2 = l1_normalization(hists2)
    sum = np.minimum(normalizedHist1,normalizedHist2).sum()
    return sum/hists1.shape[0]
    
def calculate_accuracy(query_hist:np.ndarray,support_hist:np.ndarray):
    accuracy=0
    query_ln = query_hist.shape[0]
    support_ln = support_hist.shape[0]
    for i in range(query_ln):
        mx_prob=0
        mx_ind=-1
        cur_hists = query_hist[i]
        for j in range(support_ln):
            cur_prob = compute_similarity(cur_hists,support_hist[j])
            if cur_prob > mx_prob:
                mx_prob = cur_prob
                mx_ind = j
        if i == mx_ind:
            accuracy += 1
        else:
            # print(f"False prediction: {image_names[mx_ind]}")
            pass
    accuracy /= NUM_OF_IMG
    return accuracy
def print_accuracies(support_hist,query1_hist,query2_hist,query3_hist,interval,rgb_hsv,per_channel_3d,grid_size=None):
    support_hist = np.array(support_hist)
    query1_hist = np.array(query1_hist)
    query2_hist = np.array(query2_hist)
    query3_hist = np.array(query3_hist)
    print(f"{rgb_hsv} - {per_channel_3d} - interval: {interval}")
    if grid_size != None:
        print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Query1 accuracy: {calculate_accuracy(query1_hist,support_hist)}")
    print(f"Query2 accuracy: {calculate_accuracy(query2_hist,support_hist)}")
    print(f"Query3 accuracy: {calculate_accuracy(query3_hist,support_hist)}")
    print("----------------------------")
def RGB_to_HSV_img(image:np.ndarray):
    h,w,c = image.shape
    hsv_img=np.zeros(shape=(h,w,c))
    for i in range(h):
        for j in range(w):
            hsv = RGB_to_HSV(image[i,j])
            hsv_img[i,j] = hsv
            
    return hsv_img

def convert_queries_to_hsv():
    ln,_,_,_ = query1.shape
    for i in range(ln):
        query1_hsv[i] = RGB_to_HSV_img(query1[i])
        query2_hsv[i] = RGB_to_HSV_img(query2[i])
        query3_hsv[i] = RGB_to_HSV_img(query3[i])
        supports_hsv[i] = RGB_to_HSV_img(supports[i])

def grids_by_num(image:np.ndarray,grid_size:int):
    h,w,_ = image.shape
    grid_h = h // grid_size
    grid_w = w // grid_size
    grids = []
    for i in range(grid_size):
        for j in range(grid_size):
            i_start = i*grid_h
            j_start = j*grid_w
            grid_image = image[i_start:i_start + grid_h, j_start:j_start + grid_w]        
            grids += [grid_image]
    res = np.array(grids)
    return res
    
def grid_based_run(interval,rgb_hsv,per_channel_3d,grid_size):
    bin_size = (MAX_VAL+1)//interval
    grid_num = grid_size * grid_size
    support_hist = []
    query1_hist = []
    query2_hist = []
    query3_hist = []
    support_with_grids =[]
    query1_with_grids = []
    query2_with_grids = []
    query3_with_grids = []
    if rgb_hsv == RGB: 
        for image in supports:
            grids = grids_by_num(image=image,grid_size=grid_size)
            support_with_grids += [grids]
        for image in query1:
            grids = grids_by_num(image=image,grid_size=grid_size)
            query1_with_grids += [grids]
        for image in query2:
            grids = grids_by_num(image=image,grid_size=grid_size)
            query2_with_grids += [grids]
        for image in query3:
            grids = grids_by_num(image=image,grid_size=grid_size)
            query3_with_grids += [grids]
    elif rgb_hsv == HSV:
        for image in supports_hsv:
            grids = grids_by_num(image=image,grid_size=grid_size)
            support_with_grids += [grids]
        for image in query1_hsv:
            grids = grids_by_num(image=image,grid_size=grid_size)
            query1_with_grids += [grids]
        for image in query2_hsv:
            grids = grids_by_num(image=image,grid_size=grid_size)
            query2_with_grids += [grids]
        for image in query3_hsv:
            grids = grids_by_num(image=image,grid_size=grid_size)
            query3_with_grids += [grids]
            
    if per_channel_3d == PER_CHANNEL:
        ln = np.array(support_with_grids).shape[0]
        for i in range(ln):  
            cur_hist = per_channel_color_histogram(support_with_grids[i],interval)
            support_hist += [cur_hist.reshape((3 * grid_num,bin_size))]
            
            cur_hist = per_channel_color_histogram(query1_with_grids[i],interval)
            query1_hist += [cur_hist.reshape((3 * grid_num,bin_size))]
            
            cur_hist = per_channel_color_histogram(query2_with_grids[i],interval)
            query2_hist += [cur_hist.reshape((3 * grid_num,bin_size))]
            
            cur_hist = per_channel_color_histogram(query3_with_grids[i],interval)
            query3_hist += [cur_hist.reshape((3 * grid_num,bin_size))]
               
    elif per_channel_3d == THREE_D:   
        ln = np.array(support_with_grids).shape[0]
        for i in range(ln):  
            cur_hist = threeD_color_histogram(support_with_grids[i],interval)
            support_hist += [cur_hist.reshape((grid_num,bin_size,bin_size,bin_size))]
            
            cur_hist = threeD_color_histogram(query1_with_grids[i],interval)
            query1_hist += [cur_hist.reshape((grid_num,bin_size,bin_size,bin_size))]
            
            cur_hist = threeD_color_histogram(query2_with_grids[i],interval)
            query2_hist += [cur_hist.reshape((grid_num,bin_size,bin_size,bin_size))]
            
            cur_hist = threeD_color_histogram(query3_with_grids[i],interval)
            query3_hist += [cur_hist.reshape((grid_num,bin_size,bin_size,bin_size))]
    print_accuracies(support_hist,query1_hist,query2_hist,query3_hist,interval,rgb_hsv,per_channel_3d,grid_size)        

convert_queries_to_hsv()
threed_ints = [128,64,32,16]
perc_ints = [64,32,16,8,4]

"""
Question 1 3D Color Histogram (RGB)
"""
# for i in threed_ints: 
#     threeD_run(interval=i,rgb_hsv=RGB)

"""
Question 2 3D Color Histogram (HSV)
"""
# for i in threed_ints: 
#     threeD_run(interval=i,rgb_hsv=HSV)
"""
Question 3 Per-Channel Color Histogram (RGB)
"""
# for i in perc_ints: 
#     per_channel_run(interval=i,rgb_hsv=RGB)

"""
Question 4 Per-Channel Color Histogram (HSV)
"""
# for i in perc_ints: 
#     per_channel_run(interval=i,rgb_hsv=HSV)

"""
Question 5,6,7 Grid Based Feature Extraction
Color space: HSV
Quantization interval for 3D color histogram: 64
Quantization interval for per-channel color histogram: 8
"""
# grid_sizes=[2,4,6,8]
# for i in grid_sizes:
#     grid_based_run(interval=64,rgb_hsv=HSV,per_channel_3d=THREE_D,grid_size=i)
#     grid_based_run(interval=8,rgb_hsv=HSV,per_channel_3d=PER_CHANNEL,grid_size=i)

    
end = time.time()
print(f"time: {end - start}")
