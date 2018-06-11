from core.prev_solver import CaptioningSolver
from core.prev_model import CaptionGenerator
from core.utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./resized_training_data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./resized_training_data', split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16, prev2out=True, ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=64, update_rule='adam', learning_rate=0.001, print_every=1000, save_every=10, image_path='./image/', pretrained_model=None, model_path='prev_model/lstm/', test_model='prev_model/lstm/model-20', print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    main()