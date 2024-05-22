import engineer


def main(config): 
        dataset_config = config["dataset"]
        dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

        train_loader = dataset.train_loader()
        val_loader = dataset.val_loader()
        test_loader = dataset.test_loader()

        model_config = config["model"]
        model_module = engineer.load_module(model_config.pop("module"))
        model = model_module(**model_config)

        model = model.cuda()

        optimizer_config = config["optimizer"]
        optimizer = engineer.load_module(optimizer_config.pop("module"))(
            model.parameters(), **optimizer_config
        )

        scheduler = None
        trainer_module = engineer.load_module(config["trainer"].pop("module"))

        trainer_config = config["trainer"]
        trainer_config["scheduler"] = scheduler
        trainer = trainer_module(
            **trainer_config,
        )

        trainer.fit(model, optimizer, train_loader, val_loader, test_loader)



if __name__ == "__main__":
    max_steps = [8125, 16250, 22188, 72000] # 260, 520, 710, 2295 samples
    train_size = ['_1', '_2', '_3', '']

    i = 3
    
    argv = [
        __file__,
        "-C", "configs/engineer/trainer.yaml",
        "-C", "configs/optimizer/adam.yaml",
        "-C", "configs/dataset/skeletons.yaml",
        "-C", "configs/model/o3_deh_skeletons.yaml",
        "--trainer.val_check_interval=1000",
        "--model.num_points=20",  # number of points
        "--model.output_channels=10",  # number of classes
        f"--trainer.max_steps={max_steps[i]}",
        "--dataset.batch_size=32",
        "--dataset.rot=I", # training data transformation; the default "I" means no transformation is applied
        f"--dataset.add_to_train_partition={train_size[i]}",
        f"--dataset.dset_root={'datasets'}",
        "--optimizer.lr=0.001",
        "--seed=1",
    ]

    # NOTE: you can alternatively run the file as ```python o3_skeletons.py```; 
    # for this adjust the arguments in argv above and uncomment the following

    # engineer.fire(main, argv) # then comment the following line
    engineer.fire(main)