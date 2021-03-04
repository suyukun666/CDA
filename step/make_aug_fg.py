import numpy as np
import os
import scipy.misc as misc

def run(args):

    for i in range(20):
        os.makedirs(os.path.join(args.sem_seg_out_fg_dir,str(i)), exist_ok=True)

    for seg_img in os.listdir(args.sem_seg_out_dir):
        id=seg_img.split('.')[0]
        img=misc.imread(os.path.join(args.voc12_root,'JPEGImages',id+'.jpg'))
        seg_path = os.path.join(args.sem_seg_out_dir,seg_img)
        seg=misc.imread(seg_path)
        cls_list=np.unique(seg.reshape(-1))
        if cls_list.shape[0]>2:
            continue
        mask=seg==0 
        fg_pixel=np.sum(~mask)
        ratio=fg_pixel/(mask.shape[0]*mask.shape[1])
        if ratio>0.7 or ratio<0.1:
            continue

        #裁剪
        mask=seg==0
        row=np.min(mask,axis=1)
        col=np.min(mask,axis=0)
        mn_row=np.argmin(row)
        mx_row=row.shape[0]-np.argmin(row[::-1])-1
        mn_col=np.argmin(col)
        mx_col=col.shape[0]-np.argmin(col[::-1])-1
        img=img[mn_row:mx_row+1,mn_col:mx_col+1,:]
        seg=seg[mn_row:mx_row+1,mn_col:mx_col+1]

        np.save(os.path.join(args.sem_seg_out_fg_dir,str(cls_list[1]-1),str(id)+'.npy'),
        {'img':img,'seg':seg})


    img_dic={}
    for i in range(20):
        img_dic[i]=os.listdir(os.path.join(args.sem_seg_out_fg_dir,str(i)))
    # np.save('/home/sun/papers/aug_irn/irn/voc12/seg_fg_dirs.npy',img_dic)
    np.save('voc12/seg_fg_dirs.npy',img_dic)