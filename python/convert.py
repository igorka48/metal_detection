import tensorflow as tf

# Завантаження моделі
model = tf.keras.models.load_model('metal_defect_model.keras')

# Оптимізація та конвертація моделі у TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Додаємо підтримку float16 для кращої швидкодії
tflite_model = converter.convert()

# Збереження моделі у форматі TFLite
with open('metal_defect_model.tflite', 'wb') as f:
    f.write(tflite_model)