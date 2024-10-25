from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Initialize the ImageDataGenerator with valid parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the image
img = load_img('C:\\Users\\PRIYANKA\\Desktop\\college\\genai\\dataaugmentation\\satellite.jpeg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Generate and save augmented images
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='satellite', save_format='jpeg'):
    i += 1
    if i > 20:
        break
