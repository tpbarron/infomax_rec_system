# Information Maximizing Exploration for Recommender Systems

To run the code execute:

python3 main.py --model-type BNN --epochs 5000 --tag grading --nusers 1 --use-fake-user

This will train the model and immediately start making recommendations.

python3 main.py --eta 1.0 --model-type BNN --tag grading_model --nusers 1 --use-fake-user --load-model models/grading/model_BNN_epoch_5000_retrain_0.pth

will start the recommendation process from a trained model and skip the model learning. Modifying the --eta flag will give different
recommendations. We suggest trying 1, 5, 10, 100, 500.

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

Descriptions of each file:  
main.py: main python script, used to train models, generate recommendations, and plot results.  
bnn.py: methods for building and training Bayesian network, computing KL divergence and information gain.
create_datafile.py: script for compositing the pre-processed dataset into npz file.   
datafile_ml100k.npz:  compressed file holding pre-processed dataset.
dataset.py:  script for pre-processing the MovieLens ML100K dataset, including normalization.  
movietitles.txt:  human-readable list containing titles of each movie in dataset, with corresponding index.  
simple_model.py:  simple pytorch model class, with single-layer and multi-layer options, for sanity check.  
vary-eta-fake.sh:  shell script for running system for various values of eta parameter.    
models/grading/args_snapshot.pkl:  pickle file to store arguments used in most recent execution of main.
models/grading/kls_0.1.txt:  text file storing interim recommendation results for current user.
models/grading/model_BNN_epoch_5000_retrain_0.pth:  torch save file storing model parameters.
models/grading/train.csv: CSV file storing error, accuracy information during training of model.
plotting/klfiles.zip:  archive storing most files in the plotting directory. 
plotting/kls_X.txt:   text files storing interim recommendation results for current user.
plotting/plot_X.png:  image files for plotted figures.
plotting/plot_kls.py:  python script for plotting recommendation results using matplotlib.
plotting/plot_learning.py:  python script for plotting training data using matplotlib.  
plotting/plot_tsne.py:  python script for generating t-SNE plots.
plotting/saved_rcmd_moviesX.npz:  compressed file holding recommended movie information.
plotting/tsne_recs_etaX.png:  images files for t-SNE figure.
