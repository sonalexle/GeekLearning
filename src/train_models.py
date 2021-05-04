# A script for training different pytorch lightning models

if __name__ == "__main__":

    from utils import argparser
    from models import ACPDataModule, MLP, Resnet, LinearClassifier
    import pytorch_lightning as pl
    import h5py, sys
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning import loggers as pl_loggers

    args = argparser(
        "h5trainpath", "model", "--ckpt_path", "--splitdir", "--h5testpath"
    )

    if args.model == "mlp":
        model = MLP(gamma=0.3)
    elif args.model == "resnet":
        model = Resnet(stepsize=3, learning_rate=0.003, weight_decay=1e-3)
    elif args.model == "logreg":
        model = LinearClassifier(stepsize=3, gamma=0.2)
    elif args.model == "svm":
        model = LinearClassifier(model_class="SVM", stepsize=3, gamma=0.2, hinge_deg=1)
    else:
        print("Model not supported!")
        sys.exit(1)

    if args.ckpt_path:
        model = model.load_from_checkpoint(args.ckpt_path)

    log_path = "../logs"
    tb_logger = pl_loggers.TensorBoardLogger(log_path, name=args.model, log_graph=True)

    h5trainfile = h5py.File(args.h5trainpath, "r")
    h5testfile = h5py.File(args.h5testpath, "r") if args.h5testpath else None
    dm = ACPDataModule(
        h5trainfile=h5trainfile,
        h5testfile=h5testfile,
        augment="one" if args.model == "logreg" else "all",
        datasplitdir=args.splitdir,
        num_workers=0,
        batch_size=64,
        shuffle_train=True,
    )
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=1,
        logger=tb_logger,
        callbacks=[EarlyStopping(monitor="val_acc", mode='max', patience=5)],
    )
    trainer.fit(model, dm)
    if args.model == "resnet":
        model.hparams.learning_rate = 0.005
        model.hparams.weight_decay = 1e-3
        model.set_parameter_requires_grad(freeze=False, layers=["layer4"])
        trainer = pl.Trainer(
            gpus=1,
            precision=16,
            max_epochs=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="val_acc", mode='max', patience=5)],
        )
        trainer.fit(model, dm)
    trainer.test()
    ckpt_path = f"../checkpoints/{args.model}-latest.ckpt"
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved model to {ckpt_path}")
    # Note that Tensorlogger will automatically save the model version with the best accuracy/loss.
    h5trainfile.close()
    if h5testfile:
        h5testfile.close()
