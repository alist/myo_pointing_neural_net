import coremltools

best_model_name = "./models/E_best_model.h5"

coreml_model = coremltools.converters.keras.convert(best_model_name)
coreml_model.save('./ios_model/e_myo_pointing_apple_model.mlmodel')
