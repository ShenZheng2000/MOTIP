import os
import cv2

def make_video(image_list, output_path):
    if len(image_list) == 0:
        print(f"[Skip] No images for {output_path}")
        return

    first_frame = cv2.imread(os.path.join(image_folder, image_list[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in image_list:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        if frame is None:
            continue
        video.write(frame)

    video.release()
    print("Saved:", output_path)


image_folder = "/home/shenzhen/GM_Projects/Methods/MOTIP/debug_warped/dancetrack0019_bw128_salEMA0.3"
fps = 20

base_dir = os.path.dirname(image_folder)
folder_name = os.path.basename(image_folder)

normal_video_path = os.path.join(base_dir, f"{folder_name}.mp4")
vis_video_path    = os.path.join(base_dir, f"{folder_name}_vis.mp4")
sal_video_path    = os.path.join(base_dir, f"{folder_name}_sal.mp4")

all_images = sorted(os.listdir(image_folder))

# split safely
normal_images = sorted([
    img for img in all_images
    if img.endswith(".jpg") and "_vis" not in img and "_sal" not in img
])

vis_images = sorted([
    img for img in all_images
    if img.endswith("_vis.jpg")
])

sal_images = sorted([
    img for img in all_images
    if "_sal" in img and (img.endswith(".png") or img.endswith(".jpg"))
])


make_video(normal_images, normal_video_path)
make_video(vis_images, vis_video_path)
make_video(sal_images, sal_video_path)
