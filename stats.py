import json
import numpy as np
def cal_iou_distance_mat(obj1,obj2):
    #1-iou
    obj1 = obj1.reshape(-1,2)
    obj2 = obj2.reshape(-1,2)
    inter = np.minimum(obj1[:,0].reshape(1,-1),obj2[:,0].reshape(-1,1))*np.minimum(obj1[:,1].reshape(1,-1),obj2[:,1].reshape(-1,1))
    union = (obj1[:,0]*obj1[:,1]).reshape(1,-1) + (obj2[:,0]*obj2[:,1]).reshape(-1,1) - inter
    return 1-inter/union
def cal_iou_distance(obj1,obj2):
    #1-iou
    obj1 = obj1.reshape(-1,2)
    obj2 = obj2.reshape(-1,2)
    inter = np.minimum(obj1[:,0],obj2[:,0])*np.minimum(obj1[:,1],obj2[:,1])
    union = (obj1[:,0]*obj1[:,1]) + (obj2[:,0]*obj2[:,1]) - inter
    return 1-inter/union
class kmeans(object):
    def __init__(self,vals,k=3,max_iters=200):
        self.vals = np.array(vals)
        np.random.shuffle(self.vals)
        print(self.vals.shape)
        self.dim = self.vals.shape[-1]
        self.k = k
        self.maxi = max_iters
        self.num = len(vals)
        self.terminate = False
    def initialization(self):
        assign = np.zeros(self.num,dtype=int)
        self.centers = list(range(self.k))
        k = self.k
        partn = self.num//k
        for i in range(k):
            assign[i*partn:(i+1)*partn] = i
        self.assign = assign
    def update_center(self):
        for i in range(self.k):
            if np.sum(self.assign==i)>0:
                #avoid empty cluster leading Error
                self.centers[i] = np.mean(self.vals[self.assign==i],axis=0)
        if type(self.centers) != np.ndarray:
            self.centers = np.array(self.centers)
    def cal_center_distance(self,obj1,obj2):
        return cal_iou_distance_mat(obj1,obj2)
    def cal_distance(self,obj1,obj2):
        return cal_iou_distance(obj1,obj2)
    def update_assign(self):
        self.terminate = True
        centers = self.centers
        for i in range(self.num):
            val = self.vals[i]
            tmp = self.cal_distance(val,np.stack(centers,axis=0))
            if tmp.shape[0]!=self.k:
                print(tmp.shape)
            assert tmp.shape[0]==self.k
            idx = np.argmin(tmp)
            if idx != self.assign[i]:
                self.assign[i] = idx
                self.terminate= False
    def iter(self,num):
        self.update_center()
        self.update_assign()
        if self.terminate:
            return
        else:
            if num == self.maxi:
                print("reach max iterations")
                return
            else:
                return self.iter(num+1)
    def print_cs(self):
        for i in range(self.k):
            print(round(self.centers[i,0],3),round(self.centers[i,1],3),np.sum(self.assign==i))
        centers = np.sort(np.around(self.centers,3),axis=0)        
        dis = self.cal_center_distance(centers,centers)
        print(dis[dis>1e-16].mean())
    def get_centers(self):
        centers = np.sort(np.around(self.centers,3),axis=0)  
        return [list(c) for c in centers]
    def cal_all_dist(self):
        centers = np.around(self.centers,3)[self.assign]
        distances = 1-self.cal_distance(self.vals,centers)
        print(distances.mean(),distances.min())
        return distances.mean()
    def get_cluster(self,idx):
        assert idx < self.k
        return self.vals[self.assign==idx]
class kmeans_mse(kmeans):
    def cal_distance(self,obj1,obj2):
        dis = ((obj2[:,0]/obj1[0]-1)**2).reshape(self.k,-1).sum(1)
        return dis
    def print_cs(self):
        for i in range(self.k):
            print(self.centers[i,0],np.sum(self.assign==i))
def analyze_scale_and_ratio(annos):
    allb = []
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            bw,bh = xmax-xmin,ymax-ymin
            t = 1#max(w,h)
            allb.append((bw/t,bh/t))
    scale_km = kmeans_mse(allb,k=3)
    scale_km.initialization()
    _ = scale_km.iter(0)
    for i in range(3):
        print(scale_km.centers[i][0])
        ratio_km = kmeans(scale_km.get_cluster(i))
        ratio_km.initialization()
        _ = ratio_km.iter(0)
        print("------------------")


def count_overlap(annos):
    mc = 0
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        count = np.zeros((8,8))
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            t = max(w,h)
            xc = ((xmin+xmax)/2-1)/t
            yc = ((ymin+ymax)/2-1)/t
            count[int(8*yc),int(8*xc)]+=1
        mc = max(count.max(),mc)
    print(mc)
def analyze_hw(annos):
    allb = []
    mh,mw = 1000,1000
    mxh,mxw = 0,0
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            bw,bh = xmax-xmin,ymax-ymin
            t = 1#max(w,h)
            allb.append((bw/t,bh/t))
            mh = min(mh,bh)
            mxh = max(mxh,bh)
            mw = min(mw,bw)
            mxw = max(mxw,bw)
    km = kmeans(allb,k=6,max_iters=500)
    km.initialization()
    km.iter(0)  
    print(mh,mw,mxh,mxw)
def analyze_xy(annos):
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            if ymax > h or xmax >w:
                print('???')
    print('finish')
def analyze_obj_num(annos):
    mc = 0
    for name in annos:
        num = annos[name]['obj_num']
        mc = max(num,mc)
        if mc==num:
            target = name
    print(mc,target)
def analyze_size(annos):
    res = {}
    res2 = {}
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        ts = max(w,h)
        if ts in res.keys():
            res[ts]+=1
        else:
            res[ts] = 1
        ts = round(max(h,w)/32)*32
        if ts in res2.keys():
            res2[ts]+= 1/len(annos)
        else:
            res2[ts] = 1/len(annos)
    res2 = {k: v for k, v in sorted(res2.items(), key=lambda item: item[1])}
    print(res)
    print(len(res))
    print(res2)
    print(len(res2))
if __name__ == "__main__":
    #pass
#path ='data/annotation_voc07.json'

#annos = json.load(open(path,'r'))
#analyze_hw(annos)
#img size:
#96 100 500 500
#overlap
#6
#center overlap 3(32,32)
#center overlap 4(16,16)

    anchors = [[0.06240989370204694, 0.08104391573725227],[0.1361543202763129, 0.1849733680671929],[0.19089410183808814, 0.4009908394094018],
                        [0.330588108048818, 0.4824511285878233],[0.36255652253792553, 0.2251556227577435],[0.4910133431007534, 0.5882298825023311],
                        [0.7663732637284849, 0.2674877832600824],[0.7942536132850522, 0.4810654070974923],[0.8042562250826016, 0.7227642900426251]]

    anchors = [[0.04567562487013612, 0.04877678367694419],[0.05375224461038032, 0.1059101288678919],[0.11058225513969774, 0.10200835243357394],
           [0.10076982326958268, 0.2249563017204674],[0.2121497224480688, 0.13739235134470734],[0.24658122819809095, 0.28345137753563066],
           [0.2776735214228935, 0.5156163097364138],[0.5879051949772124, 0.30330341110779224],[0.7211957658355485, 0.6396188510155153]
]

    anchors = [[0.053458141269278926, 0.07862022420023637],[0.1444091477545787, 0.11565781451235851],[0.1151994735538117, 0.24522582628542158],
           [0.31460831063222794, 0.2242885185476659],[0.21583068081593598, 0.4268351012487999],[0.38056995625914797, 0.5435495510304343],
           [0.6648272903930105, 0.3314712916726237],[0.5931101292049206, 0.7206548935846065],[0.8799995870003063, 0.5926446052236977]]
    anchors = [[0.057, 0.078], [0.102, 0.119], [0.169, 0.235], [0.246, 0.256], [0.25, 0.278], [0.447, 0.454], [0.559, 0.483], [0.791, 0.604], [0.794, 0.724]]
    anchors  =[[28.112, 38.894], [51.368, 59.001], [83.605, 117.711], [122.2, 125.505], [123.953, 139.515], [219.663, 227.468], [280.433, 237.059], [391.719, 298.828], [395.413, 361.471]]
    anchors=[[34.3, 41.965], [66.143, 110.928], [149.002, 121.749], [152.413, 174.758], [330.323, 261.792], [364.985, 331.075]]