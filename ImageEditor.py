import cv2
import numpy as np
import streamlit as st
st.title("画像加工アプリ")
st.subheader("by Kazuki Uchiyama")

select_list=["左右反転","青くなる","色反転","白黒", "顔認識", "ぴえん" ]

option = st.sidebar.selectbox(
    "行いたい加工を選んでください",
    select_list
)

uploaded_file = st.file_uploader("画像ファイルを選んでください", type=["jpg","jpeg","png","svg"])
pien = cv2.imread("pien.png", cv2.IMREAD_UNCHANGED)

if uploaded_file is not None:
    original_img = uploaded_file

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)


    st.image(original_img)
    
    btn_action = st.button("加工")
    if btn_action == True:

        if option == select_list[0] :
            reverse_img = cv2.flip(opencv_img, 1)
            st.image(reverse_img, channels="BGR")
        
        if option == select_list[1] :
            blue_img = opencv_img
            st.image(blue_img)

        if option == select_list[2]:
            color_reverse_img = cv2.bitwise_not(opencv_img)
            st.image(color_reverse_img)

        if option == select_list[3] :
            gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)   
            st.image(gray_img)
        
        if option == select_list[4]:
            gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            cascade_file = 'haarcascade_frontalface_alt.xml'
            cascade = cv2.CascadeClassifier(cascade_file)
            front_face_list = cascade.detectMultiScale(gray_img, minSize = (30, 30))

            if len(front_face_list):
                for (x,y,w,h) in front_face_list:
                    
                    cv2.rectangle(opencv_img, (x,y), (x+w, y+h), (0, 0, 255), thickness=6)
                st.image(opencv_img, channels="BGR")
            else:
                st.write("エラーです...別の画像で試してください...")


        if option == select_list[5] :
            gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            cascade_file = 'haarcascade_frontalface_alt.xml'
            cascade = cv2.CascadeClassifier(cascade_file)
            front_face_list = cascade.detectMultiScale(gray_img, minSize = (30, 30))
            if len(front_face_list):
                for (x,y,w,h) in front_face_list:
                    length = w
                    if length < h:
                        length = h
                
                    length = int(length * 1.5)
                    x = x - int((length - w) / 2)
                    y = y - int((length - h) / 2)

                    pien2 = cv2.resize(pien, dsize=(length, length), interpolation=cv2.INTER_LINEAR)
                    
                    opencv_img[y:length + y, x:length + x] = opencv_img[y:length + y, x:length + x] * (1 - pien2[:, :, 3:] / 255) + pien2[:, :, :3] * (pien2[:, :, 3:] / 255)
                    
                st.image(opencv_img, channels="BGR")
            
            else:
                st.write("エラーです...別の画像で試してください...")
    else:
        st.write("ボタンを押してください")
