import os
import argparse

import yaml
import optuna

from IPython import embed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db')
    args = parser.parse_args()

    study_name = os.path.split(args.db)[-1].split('.')[0]
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{args.db}")

    root_dpath = os.path.split(args.db)[0]
    
    best_params = study.best_params
    f = open(os.path.join(root_dpath, 'best_params.yaml'), 'w'); yaml.dump(best_params, f); f.close()
    
    fig = optuna.visualization.plot_contour(study)
    fig.update_layout(
        width=2000,
        height=2000,
    )
    fig.write_image(os.path.join(root_dpath, 'cont.png'))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(root_dpath, 'imp.png'))

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.update_layout(
        width=4000,
        height=2000,
    )
    fig.write_image(os.path.join(root_dpath, 'parallel.png'))

    embed(); exit()