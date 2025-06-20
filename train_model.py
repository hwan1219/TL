import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 경로 설정
base_dir = "C:/Users/hi/Desktop/"
train_dir = os.path.join(base_dir, "training_set/training_set")
test_dir = os.path.join(base_dir, "test_set/test_set")

# 2. 데이터 전처리
train_data_gen = ImageDataGenerator(
  rescale = 1. / 255.,
  rotation_range = 30,
  horizontal_flip = True,
  zoom_range = 0.2
)
test_data_gen = ImageDataGenerator(
  rescale = 1. / 255.
)

train_gen = train_data_gen.flow_from_directory(
  train_dir,
  target_size = (150, 150),
  batch_size = 32,
  classes=['cats', 'dogs'],
  class_mode = 'binary'
)
test_gen = test_data_gen.flow_from_directory(
  test_dir,
  target_size = (150, 150),
  batch_size = 32,
  classes=['cats', 'dogs'],
  class_mode = 'binary'
)

# 3. 사전학습 모델 불러오기 (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# 4. 커스텀 분류기 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(
  inputs = base_model.input,
  outputs = predictions
)

# 5. 모델 컴파일
model.compile(
  optimizer = Adam(learning_rate = 1e-4),
  loss = 'binary_crossentropy',
  metrics = ['accuracy']
)

model.summary()

# 6. 학습
early_stop = EarlyStopping(
  patience = 3,
  restore_best_weights = True
)

history = model.fit(
  train_gen,
  validation_data = test_gen,
  epochs = 10,
  callbacks = [early_stop]
)

# 7. 모델 저장
model.save("cat_vs_dog_model.h5")