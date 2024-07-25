import os

def rename_images(directory, prefix="car_", start_number=1):
   
    files = os.listdir(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    files.sort()
     
    for count, filename in enumerate(files):    
        new_name = f"{prefix}{str(start_number + count).zfill(2)}{os.path.splitext(filename)[1]}"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)
       
directory_path = r"D:\Car dection\dateset\train\images"
rename_images(directory_path)
