from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset


from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # Combines the arguments, model, data and experiment builders to run an experiment
    args, device = get_args()

    if any(tag in args.experiment_name for tag in ('DCML', 'MAML', 'ANIL', 'BOIL')):
        from few_shot_learning_system import MAMLFewShotClassifier
    elif any(tag in args.experiment_name for tag in ('ALFA', 'L2F')):
        from few_shot_learning_system_ALFA import MAMLFewShotClassifier
    elif any(tag in args.experiment_name for tag in ('MeTAL',)):
        from few_shot_learning_system_MeTAL import MAMLFewShotClassifier
    elif any(tag in args.experiment_name for tag in ('CxGrad',)):
        from few_shot_learning_system_ALFA import MAMLFewShotClassifier
    elif any(tag in args.experiment_name for tag in ('GAP',)):
        from few_shot_learning_system_GAP import MAMLFewShotClassifier

    # 모델을 구성한다
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(2, 3,
                                            args.image_height, args.image_width))
    maybe_unzip_dataset(args=args)

    # 데이터를 불러온다
    data = MetaLearningSystemDataLoader

    # 학습
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
