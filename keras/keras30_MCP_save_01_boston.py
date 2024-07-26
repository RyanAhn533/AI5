mcp = ModelCheckpoint(
    moonitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath)
)