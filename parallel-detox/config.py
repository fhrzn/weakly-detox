config = {
    'num_epochs': 10,
    'batch_size': 64,
    'lr': 3e-5,
    'checkpoint': 'facebook/bart-base',
    'model_path': './model/bart-parallel-full.pth'
}

inference_config = {
    'num_beams': 8,
    'do_sample': True,
    'min_length': 1,
    'max_length': 128,
    'batch_size': 64,
    'output_column_name': 'bart',
    'checkpoint': 'facebook/bart-base',
    'output_path': './data/output/bart_concat_full.txt',
    'model_path': './model/bart-parallel-full.pth'
}