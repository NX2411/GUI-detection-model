# transforms_utils.py
from PIL import Image, ImageOps

def add_padding_to_image(image, target_size=320):
    # 원본 이미지 크기 가져오기
    width, height = image.size

    # 종횡비를 유지하며 resize하기
    ratio = min(target_size / width, target_size / height)
    new_size = (int(width * ratio), int(height * ratio))
        
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
    # 패딩 계산
    delta_width = target_size - new_size[0]
    delta_height = target_size - new_size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

    # 패딩을 추가하여 종횡비 유지한 상태에서 타겟 사이즈로 만들기
    padded_image = ImageOps.expand(resized_image, padding, fill=(255, 255, 255))  # 빈 부분을 흰색으로 채움
        
    return padded_image