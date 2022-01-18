import numpy as np



def cal_alignment(output):

    align_error = 0.0

    for layout in output:
        align_error += alignment(layout)
    
    return align_error / len(output)

def cal_IoU(output):

    iou = 0.0
    for layout in output:
        iou += IoU(layout)
    
    return iou / len(output)

def cal_overlap(output):

    overlapping = 0.0

    for layout in output:
        overlapping += overlap(layout)
    
    return overlapping / len(output)


def alignment(boxes):

    '''
    input sample:
        [[class,x1,y1,x2,y2],...]
    '''
    align_error = 0.0

    if len(boxes) < 2:
        return align_error

    for i in range(len(boxes)):
        l,xc,r,t,yc,b = [],[],[],[],[],[]
        for j in range(len(boxes)):
            if i == j:
                continue
            l.append(max(min(abs(boxes[i][1] - boxes[j][1]),1.0),0.0) / 128)
            r.append(max(min(abs((boxes[i][1] + boxes[i][3]) - (boxes[j][1] + boxes[j][3])),1.0),0.0) / 128)
            t.append(max(min(abs(boxes[i][2] - boxes[j][2]),1.0),0.0) / 128)
            b.append(max(min(abs((boxes[i][2] + boxes[i][4]) - (boxes[j][2] + boxes[j][4])),1.0),0.0) / 128)
            xc.append(max(min(abs((boxes[i][3] + 2 * boxes[i][1])/2.0-(boxes[j][3]+ 2 * boxes[j][1])/2.0),1.0),0.0) / 128)
            yc.append(max(min(abs((boxes[i][4] + 2 * boxes[i][2])/2.0-(boxes[j][4] + 2 * boxes[j][2])/2.0),1.0),0.0) / 128)
        
        l,xc,r,t,yc,b = -np.log(1-min(l)), -np.log(1-min(xc)), -np.log(1-min(r)), -np.log(1-min(t)), -np.log(1-min(yc)), -np.log(1-min(b))
        align_error += min(l,xc,r,t,yc,b)
    
    return align_error


def overlap(boxes):
    '''
    input sample:
        [[x1,y1,x2,y2],...]
    '''
    overlapping = 0.0

    for i in range(len(boxes)):
        # x1 = min(boxes[i][0],boxes[i][2]) 
        # x2 = max(boxes[i][0],boxes[i][2]) 
        # y1 = min(boxes[i][1],boxes[i][3]) 
        # y2 = max(boxes[i][1],boxes[i][3])

        x1 = boxes[i][1] / 128
        x2 = boxes[i][1] + boxes[i][3]
        y1 = boxes[i][2]
        y2 = boxes[i][2] + boxes[i][4]
        s_i = (x2-x1) * (y2-y1)

        for j in range(len(boxes)):
            if i == j:
                continue
            # x3 = min(boxes[j][0],boxes[j][2]) 
            # x4 = max(boxes[j][0],boxes[j][2]) 
            # y3 = min(boxes[j][1],boxes[j][3]) 
            # y4 = max(boxes[j][1],boxes[j][3]) 

            x3 = boxes[j][1]
            x4 = boxes[j][1] + boxes[i][3]
            y3 = boxes[j][2]
            y4 = boxes[j][2] + boxes[j][4]

            w = min(x2,x4) - max(x1,x3)
            h = min(y2,y4) - max(y1,y3)

            if w<0 or h<0:
                continue
            else:
                if s_i > 0:
                    s_ij = w * h
                    overlapping += s_ij / s_i

    return overlapping


def IoU(boxes):
    '''
    mean IoU
    input sample:
        [[x1,y1,x2,y2],...]
    '''

    iou = 0.0

    for i in range(len(boxes)):
        # x1 = min(boxes[i][0],boxes[i][2])
        # x2 = max(boxes[i][0],boxes[i][2])
        # y1 = min(boxes[i][1],boxes[i][3])
        # y2 = max(boxes[i][1],boxes[i][3])
        x1 = boxes[i][1]
        x2 = boxes[i][1] + boxes[i][3]
        y1 = boxes[i][2]
        y2 = boxes[i][2] + boxes[i][4]

        for j in range(len(boxes)):

            if i == j:
                continue
            # x3 = min(boxes[j][0],boxes[j][2])
            # x4 = max(boxes[j][0],boxes[j][2])
            # y3 = min(boxes[j][1],boxes[j][3])
            # y4 = max(boxes[j][1],boxes[j][3])
            x3 = boxes[j][1]
            x4 = boxes[j][1] + boxes[i][3]
            y3 = boxes[j][2]
            y4 = boxes[j][2] + boxes[j][4]

            w = min(x2,x4) - max(x1,x3)
            h = min(y2,y4) - max(y1,y3)

            if w<0 or h<0:
                continue
            else:
                S1 = (x2-x1)*(y2-y1)
                S2 = (x4-x3)*(y4-y3)
                S_cross = w * h
                iou += S_cross/(S1+S2-S_cross + 1e-10)

    return iou / (len(boxes) + 1e-6)

if __name__ == '__main__':
    pass 

