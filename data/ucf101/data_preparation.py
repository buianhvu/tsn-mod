import glob
import os
import shutil

video_folder = "/media/data/vuba/tsn/mmaction/data/ucf101/vu_ucf/"
save_dir = "/media/data/vuba/tsn/mmaction/data/ucf101/videos/"
video_list = glob.glob(video_folder + "*.avi")
print("LEN: ", len(video_list))

for video in video_list:
    segs = video.split('_')
    folder_name = segs[2]
    video_name = video.split('/')[-1]
    sub_folder = os.path.join(save_dir, folder_name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    save_path = sub_folder + "/" + video_name
    shutil.move(video, save_path)
