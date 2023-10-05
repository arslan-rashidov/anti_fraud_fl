from joint_ml import Client

import argparse

from client_methods import model_parameters, dataset_parameters, train_parameters, test_parameters, load_model, get_dataset, train, test

output_folder = 'output/'

if __name__ == '__main__':
    client = Client(load_model_fn=load_model,
                    model_global_parameters=model_parameters,
                    dataset_global_parameters=dataset_parameters,
                    train_global_parameters=train_parameters,
                    test_global_parameters=test_parameters,
                    get_dataset_fn=get_dataset,
                    train_fn=train,
                    test_fn=test,
                    output_folder=output_folder)

    parser = argparse.ArgumentParser(description='Client Model Trainer')
    required_params = [*client.get_dataset_user_required_parameters, *client.train_user_required_parameters,
                       *client.test_user_required_parameters]

    for param in required_params:
        param_name = param['name']
        param_type = param['type']
        param_default = param['default']

        if param_type and param_default:
            parser.add_argument(param_name, type=param_type, default=param_default)
        elif param_type:
            parser.add_argument(param_name, type=param_type)
        elif param_default:
            parser.add_argument(param_name, default=param_name)
        else:
            parser.add_argument(param_name)

    args = vars(parser.parse_args())

    client.fit(**args)