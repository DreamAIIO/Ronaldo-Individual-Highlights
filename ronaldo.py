import streamlit as st
from functions import *

INPUT_VIDEO_PATH = "./input_vid" 
SUB_VIDS_PATH = "./sub_vids"
HIGH_VIDS_PATH = "./high_vids"

#this will create 3 folders on the paths defined above
os.makedirs(INPUT_VIDEO_PATH, exist_ok=True)
os.makedirs(SUB_VIDS_PATH, exist_ok=True)
os.makedirs(HIGH_VIDS, exist_ok=True)

st.title("Ronaldo Detection")
st.write("Upload a video")
uploaded_file = st.file_uploader("File Uploader for Ronaldo highlights",type=['mp4', 'avi'])


if uploaded_file != None:
    with open(os.path.join(INPUT_VIDEO_PATH,uploaded_file.name),"wb") as f:    
        f.write(uploaded_file.getbuffer())

    src_vid = get_files(INPUT_VIDEO_PATH)
    st.write(src_vid)
    vid = src_vid[0]
    st.write(vid)
    file_stats = os.stat(vid)
    st.write(file_stats.st_size / (1024 * 1024))

    video_file = open(vid, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    

    
if st.button("Click here to make Ronaldo Highlights"):

        if uploaded_file is not None:
            model = YOLO('./ron_3k.pt') #these are custom model weights, you can load your own model weights
            st.write(model.names) #Just to make sure at which index the 'ronaldo' class is placed, in this case it's 1
            fps = get_fps(vid)
            split_vid(vid, SUB_VIDS_PATH)
            split_vids = get_files(SUB_VIDS_PATH)
            split_vids = sort_list(split_vids)
            x=0
            for sv in split_vids:
                #you can play around with the conf (confidence) value
                results = model.track(source=sv, persist=True, classes=1,
                          conf=0.65, tracker="bytetrack.yaml", save=False, show=False,
                          verbose=False, save_txt=False) 
                results = filter_vid(results,fps)
                create_vid(results, HIGH_VIDS_PATH, x, fps)
                x = x+1
            concatenate_videos(HIGH_VIDS_PATH, fps, "./finalVid.mp4")
            
            t_video_file = open("./finalVid.mp4", 'rb')
            t_video_bytes = t_video_file.read()
            st.video(t_video_bytes)
            with open("./finalVid.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="Final.mp4",
                        mime="video/mp4"
                      )
  
        else:
            st.write("Please, upload a video first")
    
