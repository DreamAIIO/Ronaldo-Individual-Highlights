import os
import gc
import torch
from pathlib import Path
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import glob
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sort_list(paths_list):
    paths_list = [str(v) for v in paths_list]
    paths_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return paths_list
    

def setify(o):
    return o if isinstance(o, set) else set(list(o))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path, extensions=None, recurse=True, folders=None, followlinks=True, make_str=False
):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    if folders is None:
        folders = list([])
    path = Path(path)
    if extensions is not None:
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    if make_str:
        res = [str(o) for o in res]
    return list(res)

def del_fol (file_path) :
    folder = file_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            

def filter_vid(results,fps):
    imgs=[]
    i=0
    fps = int(fps)
    step = int(fps*2)
    while i < len(results):
        #img = results[i].orig_img
        cls = list(results[i].boxes.cls)
        if cls != []:
            temp = []
            c=0
            for x in range(i,i+fps):
                if i+fps > len(results):
                    break
                temp.append(results[x].orig_img)
                t_cls = list(results[x].boxes.cls)
                if t_cls == []:
                    c=c+1
            if c <= 15:
                for f in range(i+fps, i+step):
                    if i+fps > len(results) or i+step > len(results):
                        break
                    temp.append(results[f].orig_img)
                imgs+=temp
                i = i+step
            else:
                i = i+fps
        else:
            i = i+1
            
    return imgs
            
def create_vid(img_list, dest_path, x, fps):
    size = (720, 1280)
    vid_name = dest_path+'wowTest'+str(x)+'.mp4'
    temp = 'wowTest'+str(x)+'.mp4'
    out = cv2.VideoWriter(vid_name,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,(size[1],size[0]),
                          isColor=True)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()
    #os.system("ffmpeg -i highVids/wowTest0.mp4 -vcodec libx264 compVids/wowTest0.mp4")
    
    
def split_vid(video, split_path):
    video = VideoFileClip(str(video))
    half = (video.duration / 2.0) - 6
    split = half / 10.0
    c = 0
    i = 0
    #clips = []
    while i < half:
        if (i+split) > half:
            #diff = abs((i+split) - half)
            sub_clip = video.subclip(i,half)
            try:
                sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            except IndexError:
                try:
                    clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264",
                                        audio=False)
                except Exception as e:
                    logger.info("exception caught: %s" % e)
            #sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            i = i+split
            c = c+1
        else:
            sub_clip = video.subclip(i,i+split)
            sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            #clips.append(sub_clip)
            i = i+split
            c = c+1
        
def get_fps(video):
    video = VideoFileClip(str(video))
    fps = video.fps
    return fps
        
def final_vid(dest_path):
    highVids = get_files(dest_path)
    highVids = sort_list(highVids)
    for i in range(len(highVids)):
        highVids[i] = VideoFileClip(highVids[i])
    video = concatenate_videoclips(highVids)
    video.write_videofile("./final.mp4", codec="libx264")
    
def concatenate_videos(dest_path, fps, new_video_path):
    highVids = get_files(dest_path)
    #li = sort_list(li)
    for i in range(len(highVids)):
        highVids[i] = './'+str(highVids[i])
    highVids.sort()
    size = (720, 1280)
    video = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, (size[1],size[0]))

    for v in range(len(highVids)):
        curr_v = cv2.VideoCapture(highVids[v])
        while curr_v.isOpened():
            r, frame = curr_v.read()
            if not r:
                break
            video.write(frame)

    video.release()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
