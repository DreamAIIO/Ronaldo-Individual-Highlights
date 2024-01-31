import streamlit as st
from functions import *

st.title("Ronaldo Detection")
st.write("Upload a video")
uploaded_file = st.file_uploader("File Uploader for Ronaldo highlights",type=['mp4', 'avi'])
src_path = "./input_vids/"
dest_path = "./highVids/"
split_path = "./testingVids/"


if uploaded_file != None:
    with open(os.path.join(src_path,uploaded_file.name),"wb") as f:    
        f.write(uploaded_file.getbuffer())

    src_vid = get_files(src_path)
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
            model = YOLO('./ron_3k.pt')
            st.write(model.names)
            fps = get_fps(vid)
            split_vid(vid, split_path)
            split_vids = get_files(split_path)
            split_vids = sort_list(split_vids)
            x=0
            for sv in split_vids:
                results = model.track(source=sv, persist=True, classes=1,
                          conf=0.7, tracker="bytetrack.yaml", save=False, show=False,
                          verbose=False, save_txt=False)
                results = filter_vid(results,fps)
                create_vid(results, dest_path, x, fps)
                x = x+1
            concatenate_videos(dest_path,"./finalVidYo.mp4")
            
            t_video_file = open("./final.mp4", 'rb')
            t_video_bytes = t_video_file.read()
            st.video(t_video_bytes)
            with open("./final.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="tempFinal.mp4",
                        mime="video/mp4"
                      )
  
        else:
            st.write("Please, upload a video first")
    
