# Information Maximizing Exploration for Recommender Systems

To run the code execute.

python3 main.py --model-type BNN --epochs 5000 --tag grading --nusers 1 --use-fake-user

This will train the model and immediately start making recommendations.

python3 main.py --eta 1.0 --model-type BNN --tag grading_model --nusers 1 --use-fake-user --load-model models/grading/model_BNN_epoch_5000_retrain_0.pth

will start the recommendation process from a trained model and skip the model learning. Modifying the --eta flag will give different
recommdations. We suggest trying 1, 5, 10, 100, 500.

Arguments:
  --model-type: FC or BNN, default=BNN
  --epochs: training epochs, default=5000
  --eta: expl param, default=0.1
  --load-model: path to pytorch model to load
  --user: Which user to train on, default=1
  --nusers: num users to use, default=5
  --use-kernel: use polynomial kernal (store_true)
  --use-non-lin: use nonlinear model (store_true)
  --use-fake-user: use hand designed user for testing (store_true)
  --use-default-recs: recommend by highest prob, no info gain (store_true)
  --use-user-tag: Append user information to movie features. Needed when more than 1 user. Note: not compatible with fake user (store_true)
  --vpi: Assume user doens't like as proxy. (store_true)
  --tag: Folder name to create in models/ to save exp data, default=tmp

Dependencies:
python3
pytorch (needs to be installed separately: http://pytorch.org/)
numpy
sklearn
