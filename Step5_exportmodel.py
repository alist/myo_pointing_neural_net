import coremltools

best_model_name = "./models/I_model.h5"

coreml_model = coremltools.converters.keras.convert(best_model_name)
coreml_model.save('./ios_model/IMyoPointingModel.mlmodel')
