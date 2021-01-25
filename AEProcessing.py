import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate as interpolate_sci
from datetime import datetime, timedelta
import urllib.request 
import os
#import warnings as w
#w.filterwarnings(action = 'ignore') 

def grab(date):
    '''
    Get downloaded date file from 
    '''
    img = cv2.imread('rtae_'+date+'.png')
    return img

def show(img):
    '''
    Display the image 
    '''
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)

def crop(img):

    '''
    Crop the image
    '''

    x1 = 78
    x1 = 80
    y1 = 24
    y1 = 27
    x2 = 650
    y2 = 199
    #y2 = 500
    crop_img = img[y1:y2, x1:x2]
    return crop_img
#y: -2000, 1000
#x: 0, 24
#zero at 41

def rm_greylines(img):
    '''
    The original image has grey gridlines. This function filters them out. 
    '''
    lower = np.array([120, 120, 120], np.uint8)
    upper = np.array([191, 191, 191], np.uint8)
    greylines = cv2.inRange(img, lower, upper)
    notgreylines = cv2.bitwise_not(greylines)
    lower = np.array( [240, 240, 240], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    mask = cv2.inRange(img, lower, upper)

    mask = cv2.bitwise_not(mask)
    mask = mask*notgreylines
    img = cv2.bitwise_and(img, img, mask = mask)
    return img

def rm_color(img):
    '''
    The image dataset is a multi-dimensional array with RGB (or GBR depending on how I loaded it, can't remember) values. 
    This masking function takes all the data points above [0, 0, 0] and replaces it with a 2-d [255] value. 
    CHECK ME 
    '''
    lower_black = np.array( [1, 1, 1], np.uint8)
    upper_white = np.array( [255, 255, 255], np.uint8)
    colorless_img = cv2.inRange(img, lower_black, upper_white)
    return colorless_img

'''
This is how I calculated the scaling values. 
Y-scaling
    0-41, 1000
    41-147, -2000
    scale value: 2000/(147-41) = 18.867924528301888
X-scaling
    0-610, 24
    scale value: 24/610 = 0.04203152364273205
'''


def transform_values(mask, zero):
    '''
    Parameters
    ----------
    mask : ARRAY
        The boolean values of the mas.
    zero : INT
        The element in the array that corresponds to the zeroth value.

    Returns
    -------
    valuelist : ARRAY
        As it stands, the dataset's values are [0, 0] at the top left corner. 
    This transforms that data so that the zero line is where the zero line is 
    scaled in the array we export. 
    '''
    
    valuelist = []
    indices = [list(range(len(mask[0])))]*len(mask)
    for i in range(len(indices)):
        for j in range(len(indices[0])):
            if mask[i][j] > 0:
                #valuelist += [[indices[i][j], i]]
                valuelist += [[indices[i][j], zero-i]]    
    return valuelist

    '''
    valuelist = []
    indices = [list(range(len(mask[0])))]*len(mask)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 0:
                #valuelist += [[indices[i][j], i]]
                valuelist += [[j, zero-i]]
                
    
    return valuelist 
    '''

def export(final_data, date):
    '''
    Exports final image. 
    '''
    final_data.to_csv(date+'_ProxyAE.csv')
    #np.savetxt(date+'OpenCV.csv', valuelist, fmt = "%f, %f", delimiter = ',')

def contours(colorless_img, img):
    '''
    New way to obtain image. The contour function takes just the outline of the data. 
    '''
    ret, thresh = cv2.threshold(colorless_img, 254, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contour = cv2.drawContours(np.zeros_like(img), contours, -1, (255,255,255), 1)
    lower = np.array( [240, 240, 240], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    mask = cv2.inRange(final_contour, lower, upper)
    return mask

def sort(values):
    '''
    This comes in handy for graphing, but isn't totally necessary. It just sorts the values. 
    '''
    sorted_values = values[np.argsort(np.asarray(values)[::, 0])]
    sorted_values = sorted_values.astype('f')
    return sorted_values

def scale(sorted_values, x_scale, y_scale):
    '''
    This scales the values. You might want to look at a better way to do this. 
    '''
    #y
    sorted_values[::, 1] = sorted_values[::, 1]*y_scale
    #x
    sorted_values[::, 0] = sorted_values[::, 0]*x_scale
    return sorted_values

def analysis(values_dic, date, savefig):
    '''
    Separates, interpolated, and calculates rolling average of the AE data. 
    Inputs: Values as a dictionary, the date for file naming, and whether you want to save the figure. 
    Returns: Data as a dictionary. 
    '''

    data = pd.DataFrame(data=values_dic)
    data = data.loc[(data['X'] >= 0.) & (data['X'] <= 24.)]
    #data = data[~data['X'].duplicated()] #use this line to find any duplicated x values... remove if mostly duplicates.
    mask = data['AL/AU'] < 0
    data['AU'] = data['AL/AU'].mask(mask)
    data['AL'] = data['AL/AU'].mask(~mask)
    Time = np.linspace(0,24,1440)

    #plt.plot(data['AU'], '.')
    #plt.plot(data['AL'], '.')
    #plt.title("OpenCV (NEW)")
    #max(-data['AL'])

    AL_nans = np.ma.masked_invalid(data['AL'].values) #singlin out the nans
    AU_nans = np.ma.masked_invalid(data['AU'].values)
    f_AL = interpolate_sci.interp1d(data['X'].values[~AL_nans.mask], data['AL'].values[~AL_nans.mask],bounds_error = False, fill_value = (data['AL'].values[~AL_nans.mask][0], data['AL'].values[~AL_nans.mask][-1]))
    f_AU = interpolate_sci.interp1d(data['X'].values[~AU_nans.mask], data['AU'].values[~AU_nans.mask],bounds_error = False, fill_value = (data['AU'].values[~AU_nans.mask][0], data['AU'].values[~AU_nans.mask][-1]))
    #f_AU = interpolate_sci.interp1d(data['X'].values[~AU_nans.mask], data['AU'].values[~AU_nans.mask], fill_value = 'extrapolate')
    #f_AL = interpolate_sci.interp1d(data['X'].values[~AL_nans.mask], data['AL'].values[~AL_nans.mask], fill_value = 'extrapolate')
    AL_new = f_AL(Time)
    AU_new = f_AU(Time)
    AE = AU_new - AL_new

    #Storing Time, interpolated AL and AU, and calculated AE data (line 34) into a new dataframe, then including the 5 min center average of AE
    interpolated_data = pd.DataFrame({'Time':Time * 60.0, 
                                      'AL_interp':AL_new, 
                                      'AU_interp':AU_new, 
                                      'AE': AE})

    interpolated_data['AE_avg'] = interpolated_data['AE'].rolling(5, center = True).mean()
    

    #Displaying Result
    fig, testplt = plt.subplots()
    testplt.plot(interpolated_data['Time'], interpolated_data['AE_avg'], '-r')
    testplt.set_title('OpenCV')
    testplt.set(xlim=(0, 1500), ylim=(-500, 2000))
    testplt.set_ylabel('AE')
    plt.grid()
    testplt.set_aspect(aspect = 0.2)
  
    if savefig:
        plt.savefig(date+'_OpenCVExample.png')
        
    return interpolated_data

def get_AE(date, savefig):
    '''
    Puts all of the above functions together in a procedure. Give it a date and it will give you the AE data. 
    '''
    y_scale = 2000/(147-41)
    x_scale = 24/570
    zero = 55
    cropped = crop(grab(date))
    img = rm_greylines(cropped)
    colorless_img = rm_color(img)
    contour_mask = contours(colorless_img, img)
    #show(contour_mask)

    values = np.asarray(transform_values(contour_mask, zero))
    
    sorted_values= sort(values)
    sorted_values = sorted_values[sorted_values[::, 0] > 1]

    scaled_values = scale(sorted_values, x_scale, y_scale)

    values_dic = {'X': scaled_values[::, 0], 'AL/AU': scaled_values[::, 1]}

    interpolated_data = analysis(values_dic, date, savefig)
    return interpolated_data

def main(enddate = str(datetime.now().strftime("%Y%m%d")), days = 31, savefig = False):
    '''

    !!!! Change the directories !!!!


    Main function. The default is: whenever the file is run, scrape and process 31 days from the current date. It will not save the figure. 
    The files will be named: date+_ProxyAE.csv, e.g. 20201123_ProxyAE.csv
    '''

    ####
    finalproduct_dir = r"C:\Users\lauraigs\Documents\Research\AE_Digitization\2020\OpenCVProxies"
    ####
    os.chdir(finalproduct_dir)
    enddate = datetime.strptime(enddate, '%Y%m%d')
    for day in reversed(range(days)):
        td = timedelta(days = day)
        currentdate = enddate-td
        datecode = str(currentdate.strftime("%Y%m%d"))
        fname = 'rtae_'+str(datecode)+'.png'
        suffix = str(datecode)[:6]+'/'+fname
        url = 'http://wdc.kugi.kyoto-u.ac.jp/ae_realtime/'+suffix
        urllib.request.urlretrieve(url, fname)
        print(datecode, " downloaded")
        final_data = get_AE(datecode, savefig)
        export(final_data, datecode)
        os.remove(fname) 
        print(datecode, " proxy AE generated")
        
        
    return

if __name__ == '__main__':
   main()
