import newton

builder = newton.ModelBuilder()
builder.add_usd("objects/scene03/scene01.usdc")

model = builder.finalize()

viewer = newton.viewer.ViewerUSD(output_path="static.usd")
viewer.set_model(model)
viewer.begin_frame(0.0)
viewer.log_state(model.state())
viewer.end_frame()
viewer.close()
