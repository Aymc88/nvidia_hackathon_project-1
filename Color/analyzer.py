from PIL import Image

def analyze_image_sentiment(image_file):
    """
    V1基础版：分析上传的图片文件中的红绿像素占比。
    """
    try:
        # 重置文件指针以允许重复读取
        image_file.seek(0)
        image = Image.open(image_file).convert('RGB')
        width, height = image.size
        red_pixels = 0
        green_pixels = 0
        
        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                is_red = r > 150 and g < 100 and b < 100
                is_green = g > 150 and r < 100 and b < 100
                if is_red:
                    red_pixels += 1
                elif is_green:
                    green_pixels += 1
        
        total_colored_pixels = red_pixels + green_pixels
        if total_colored_pixels == 0:
            return 0.5, 0, 0

        sentiment_score = green_pixels / total_colored_pixels
        return sentiment_score, green_pixels, red_pixels
    except Exception as e:
        print(f"V1图像处理出错: {e}")
        return None, 0, 0

def analyze_image_sentiment_v2(image_file):
    """
    V2智能版：使用时间加权分析红绿像素占比。
    """
    try:
        # 重置文件指针以允许重复读取
        image_file.seek(0)
        image = Image.open(image_file).convert('RGB')
        width, height = image.size
        
        weighted_red_score = 0
        weighted_green_score = 0
        
        for y in range(height):
            for x in range(width):
                weight = (x + 1) / width
                r, g, b = image.getpixel((x, y))
                is_red = r > 150 and g < 100 and b < 100
                is_green = g > 150 and r < 100 and b < 100

                if is_red:
                    weighted_red_score += weight 
                elif is_green:
                    weighted_green_score += weight
        
        total_weighted_score = weighted_red_score + weighted_green_score
        if total_weighted_score == 0:
            return 0.5, 0, 0

        sentiment_score = weighted_green_score / total_weighted_score
        return sentiment_score, weighted_green_score, weighted_red_score
    except Exception as e:
        print(f"V2图像处理出错: {e}")
        return None, 0, 0