import coremltools

best_model_name = "./models/K_model.h5"

coreml_model = coremltools.converters.keras.convert(best_model_name)
coreml_model.save('./ios_model/KMyoPointingModel.mlmodel')
