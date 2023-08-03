import streamlit as st
import subprocess
import os
from PIL import Image
import tempfile
# Khách hàng vào trang, sẽ có ô để tải ảnh lên, 1 ô để chọn style, 1 ô để show kết quả
# 1 nút để tải hình ảnh về
# Thêm file
#Chạy chương trình: streamlit run make_streamlit.py

# # Bước 1: Tạo ra fodel chứa tất cả các hình ảnh chạy thử nghiệm với path
# model_load_path = 'result/'
# test_content_path_mau = 'imgs/contents/000000000569.jpg'
# output_path_mau = 'image_mau/'
# model_load_mau_paths = [os.path.join(model_load_path, filename) 
#                for filename in os.listdir(model_load_path) 
#                if filename.lower().endswith(('.pth'))]
# for i in model_load_mau_paths:
#     print(i)
#     index_to_slicing = i.rfind('/')
#     name_save = i[index_to_slicing:-4] + '.png'
#     print(name_save)
#     command_mau = f"python test_main.py --model_load_path {i} --test_content {test_content_path_mau} --output {output_path_mau}/{name_save}"
#     process_mau = subprocess.Popen(command_mau, stdout=subprocess.PIPE, shell=True) 
#     stdout, stderr = process_mau.communicate()
#     print(stdout.decode("utf-8"))

## Bước 2 chia làm 3 cột
# col1, khách hàng úp ảnh muốn transfer
col1, col2, col3= st.columns(3)
file = []
with col1:
    file = st.file_uploader('chọn file')
    if file is not None:
        st.image(file)
#st.text(type(file))
#Col 2: Show ra các style ảnh để khách hàng chọn style Chọn file style
selected_image = ''
with col2:
    st.title("Chọn ảnh yêu thích")

    image_folder = "image_mau"
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    selected_image = st.selectbox("Chọn ảnh:", image_files)
    #selected_image nó là đường dẫn đến hình ảnh demo
    st.image('imgs/contents/000000000569.jpg', caption='Đây là hình ảnh gốc') # show ra hình ảnh ban đầu để so sánh với hình ảnh sau test

    if selected_image:
        image_path = os.path.join(image_folder, selected_image)
        st.image(image_path, caption=selected_image, use_column_width=True)
        print(selected_image[selected_image.rfind('/'):-4])
        model_load_paths = 'result'
        
        for i in os.listdir(model_load_paths):
            if selected_image[selected_image.rfind('/'):-4] == i[i.rfind('/'):-4]:
                selected_image = i
                break
check = False
result = []
#Thực hiện generate và show ra 
with col3:

    st.title("Chạy Mô Hình với Streamlit")

    model_load_path = selected_image

    desired_temp_dir = "D:/A_HIT_AI/Deep_Learning/HIT_Product/test_result"
    temp_dir = os.path.join(desired_temp_dir)
    # Tạo thư mục tạm thời ở vị trí mong muốn
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, "temp_image.png")
    with open(temp_image_path, 'wb') as temp_file:
        temp_file.write(file.read())
    test_content_path = temp_image_path 
        
    output_path = "test_result"

    if st.button("Chạy"):
        check = True
        command = f"python test_main.py --model_load_path result/{model_load_path} --test_content {test_content_path} --output {output_path}/tt.png"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        result = f'{output_path}/tt.png'
        st.write(output.decode("utf-8"))
        st.image(result, caption='Hình ảnh kết quả', use_column_width=True)
        st.success("Hoàn thành!")

# TẢi file
with open('test_result/tt.png', 'rb') as file_img:
    if check == True:
        st.download_button(
            label="Tải ảnh",
            data=file_img,
            file_name = 'result.png',
            mime="image/png"
        )

