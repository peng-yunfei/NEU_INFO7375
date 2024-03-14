class Dataset:
    def __init__(self):
        self.images = []
    
    def add_image(self, image):
        self.images.append(image)
    
    def get_images(self):
        return self.images
    
    def get_size(self):
        return len(self.images)

class DatasetSplit:
    def __init__(self, train_ratio, validation_ratio, test_ratio):
        self.training_set = Dataset()
        self.validation_set = Dataset()
        self.testing_set = Dataset()

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

    def add_image(self, image):
        total = self.training_set.size() + self.validation_set.size() + self.testing_set.size() + 1
        train_threshold = self.train_ratio * total
        validation_threshold = train_threshold + (self.validation_ratio * total)
        
        if self.training_set.size() < train_threshold:
            self.training_set.add_image(image)
        elif self.validation_set.size() < validation_threshold - train_threshold:
            self.validation_set.add_image(image)
        else:
            self.testing_set.add_image(image)
    
    def get_training_set(self):
        return self.training_set.get_images()
    
    def get_validation_set(self):
        return self.validation_set.get_images()
    
    def get_testing_set(self):
        return self.testing_set.get_images()
