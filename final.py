import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os
class face:
    def __init__(self,image_path):
        self.symmetry_df = pd.read_csv('landmarks.csv')
        self.symmetry_df = self.symmetry_df.dropna()
        self.ratios_df = pd.read_csv('ratios.csv')
        self.angles_df = pd.read_csv('angles.csv')
        self.symmetry_measurements = {
            'jaw flare': ['RJA','LJA'],
            'brow upper': ['rightEyebrowUpper','leftEyebrowUpper'],
            'brow lower': ['rightEyebrowLower','leftEyebrowLower'], 
            'mouth inner': ['lipsUpperInner','lipsLowerInner'],
            'mouth outer': ['lipsUpperOuter','lipsLowerOuter'],
            'eye upper 0': ['rightEyeUpper0','leftEyeUpper0'],
            'eye upper 1': ['rightEyeUpper1','leftEyeUpper1'],
            'eye upper 2': ['rightEyeUpper2','leftEyeUpper2'],
            'eye upper 3': ['rightEyeUpper3','leftEyeUpper3'],
            'nostril': ['nostrilRight','nostrilLeft']
            }
        self.img = cv2.imread(image_path)
        self.h, self.w = self.img.shape[:2]
        self.mp_face_mesh = mp.solutions.face_mesh
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  
            min_detection_confidence=0.5
        ) as face_mesh:
            self.results = face_mesh.process(self.img)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=2)
        face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        print("Faces detected:", bool(self.results.multi_face_landmarks))
        self.dic = {}
        h, w = self.img.shape[:2]  # Get image dimensions
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                for i,lm in enumerate(face_landmarks.landmark):
                    self.dic[str(i)] = np.array([int(lm.x * self.w), int(lm.y * self.h)])

    def create_landmarks(self,l=None):
        temp_img = cv2.resize(self.img, (480, 640))
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=temp_img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )
                if l != None:
                    for i in l:
                        lm = face_landmarks.landmark[i]
                        x = int(lm.x * self.w)
                        y = int(lm.y * self.h)
                        cv2.circle(temp_img, (x, y), radius=2, color=(0, 255, 0), thickness=4)       
        cv2.imshow("Selected Landmarks", temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transform(self,name):
        x_arr = []
        y_arr = []
        arr1 = []
        arr2 = []
        cols = self.symmetry_measurements[name]
        points_R = self.symmetry_df[cols[0]]
        points_L = self.symmetry_df[cols[1]]
        for i in range(0,len(points_R)):
            arr1.append(self.dic[str(int(points_R[i]))])
            arr2.append(self.dic[str(int(points_L[i]))])
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        if 'mouth' in name:
            mid = [61, 146, 91, 181, 13, 311, 321, 375, 291]
        else:
            mid = [10,151,6,4,2,13,14,152]
        for face_landmarks in self.results.multi_face_landmarks:
            for idx in mid:
                lm = face_landmarks.landmark[idx]
                x_arr.append(lm.x*self.w)
                y_arr.append(lm.y*self.h)
        c,m = np.polyfit(x_arr,y_arr,1)
        if 'mouth' in name:
            difference = np.array([0,y_arr[0]])
        else:
            difference = np.array([x_arr[0],0])
        arr1 = (arr1 - difference).T
        arr2 = (arr2 - difference).T
        theta = np.arctan(m)
        M = np.array([[np.cos(2*theta),np.sin(2*theta)],
                      [np.sin(2*theta),-np.cos(2*theta)]])
        new = np.matmul(M,np.array(arr1))
        score_arr = np.array(arr2)-new
        score_arr = (score_arr[0])**2+(score_arr[1])**2
        score = np.mean(score_arr)
        score = np.exp(-1e-6*score)
        return score

    def theta(self,col):
        l = self.angles_df[col]
        m = np.array([self.dic[str(i)] for i in l])
        return 180 - (np.arccos(np.dot(m[0]-m[1],m[1]-m[2]) / (np.linalg.norm(m[0]-m[1])*np.linalg.norm(m[1]-m[2]))) * 180/np.pi)

    def distance(self,name):
        '''calculates distance ratios
           some ratios have custom calculations'''
        def alt1():
            return (np.linalg.norm(self.dic['33']-self.dic['133'])+np.linalg.norm(self.dic['362']-self.dic['263']))/(np.linalg.norm(self.dic['159']-self.dic['145'])+np.linalg.norm(self.dic['386']-self.dic['374']))
        def alt2():
            return 2*np.linalg.norm(self.dic['223']-self.dic['159'])/np.linalg.norm(self.dic['159']-self.dic['145'])
        def alt3():
            return np.linalg.norm(self.dic['2']-self.dic['200'])/np.linalg.norm(self.dic['9']-self.dic['2'])
        def alt4():
            return np.linalg.norm(self.dic['234']-self.dic['454'])/5
        def alt5():
            return 2*np.linalg.norm(self.dic['468']-self.dic['473'])/(np.linalg.norm(self.dic['468']-self.dic['0'])+np.linalg.norm(self.dic['473']-self.dic['0']))
        names = {
            'PFL-PFH': alt1(),
            'eyebrow': alt2(),
            'golden': alt3(),
            'fifths': alt4(),
            'midface': alt5()
        }
        if name in names.keys():
            return names.get(name)
        else:
            l = self.ratios_df[name]
            m = np.array([self.dic[str(i)] for i in l])
            return np.linalg.norm(m[0]-m[1])/np.linalg.norm(m[2]-m[3])
        
def facial_convexity_angle(img_R,img_L):
    angle_R = face(img_R).theta('FCA')
    angle_L = face(img_L).theta('FCA')
    return (angle_L +angle_R)/2


def run_all(image_path,image_path_L=None,image_path_R=None,output=False):
    '''runs all methods in the face class and returns a dictionary of results
       pass image_path_L and image_path_R to calculate facial convexity angle
       and output=True to print results to terminal'''
    if '_smile' in image_path:
        string = ' smile'
    else:
        string = ''
    metrics = {}
    # create face object
    obj1 = face(image_path)
    # draw landmarks on image
    obj1.create_landmarks()
    # get list of ratios, angles, and symmetry measurements from csv files
    ratios = pd.read_csv('ratios.csv').columns.tolist()
    angles = pd.read_csv('angles.csv').columns.tolist()
    symmetry = pd.read_csv('landmarks.csv').columns.tolist()
    #saves results to dictionary and prints to terminal if output=True
    for i in ratios:
        val = obj1.distance(i)
        if output == True:
            print(f'{i} ratio: {val}')
        metrics[str(i)+' ratio'+string] = val
    for i in angles:
        val = obj1.theta(i)
        if output == True:
            print(f'{i} angle: {val}')
        metrics[str(i)+' angle'+string] = val
    #for i in symmetry:
    #    print(f"{i} symmetry: {obj1.transform(i)}")
    if image_path_L != None:
        #print(image_path_L,image_path_R)
        val = facial_convexity_angle(image_path_R,image_path_L)
        print(f'Facial Convexity Angle: {val}')
        metrics['FCA'] = val
    return metrics

def save_results(output=False):
    '''save results for each image in the same csv file,
       uses the run_all function to run each method in the object,
       assumes images are named image1.jpg, image2.jpg, etc.
       each set of images should have 4 images: imageX.jpg, imageX_L.jpg, imageX_R.jpg, imageX_smile.jpg
    '''
    # counts the number of sets of images 
    count = 0
    for file in os.listdir('images'):
        count += 1
    count = int(count/4)

    l = []
    # loops through each set of images
    for num in range(1,count+1):
        # makes the path for each image in the setusing the naming convention
        path = 'images/image'+str(num)+'.jpg'
        path_L = 'images/image'+str(num)+'_L.jpg'
        path_R = 'images/image'+str(num)+'_R.jpg'
        path_smile = 'images/image'+str(num)+'_smile.jpg'

        # get metrics for the smiling and normal faces and save it to a dictionary
        dic1 = run_all(path,image_path_L=path_L,image_path_R=path_R,output=output)
        dic2 = run_all(path_smile,output=output)
        dic_total = dic1 | dic2
        l.append(dic_total)

    # saves the list of dictionaries to a csv file
    df = pd.DataFrame(l)
    df.to_csv('results.csv',index=False)

# pass output=True to print results to terminal
save_results()