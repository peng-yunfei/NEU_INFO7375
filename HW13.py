class Image:
    def __init__(self, data):
        self.data = data 
        self.width = len(data[0])
        self.height = len(data)
        
class Filter:
    def __init__(self, data):
        self.data = data  
        self.size = len(data)

def convolve(image, filter):
    output_height = image.height - filter.size + 1
    output_width = image.width - filter.size + 1

    output = [[0 for _ in range(output_width)] for _ in range(output_height)]

    for i in range(output_height):
        for j in range(output_width):
            sum = 0
            for k in range(filter.size):
                for l in range(filter.size):
                    sum += image.data[i + k][j + l] * filter.data[k][l]
            output[i][j] = sum

    return output

original_image_data = [
    [10, 20, 30, 40, 50, 60],
    [20, 30, 40, 50, 60, 70],
    [30, 40, 50, 60, 70, 80],
    [40, 50, 60, 70, 80, 90],
    [50, 60, 70, 80, 90, 100],
    [60, 70, 80, 90, 100, 110]
]


filter_data = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

image = Image(original_image_data)
filter = Filter(filter_data)

convolved_image = convolve(image, filter)
print(convolved_image)
